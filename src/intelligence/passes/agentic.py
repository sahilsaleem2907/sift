"""Bounded agentic review loop with tool-calling (Phase 4, high effort)."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from litellm import acompletion

from src import config
from src.intelligence.ast.function_extract import FunctionChunk
from src.intelligence.capability import ModelCapability
from src.intelligence.effort import EffortPlan
from src.intelligence.llm_client import (
    REVIEW_FILE_SYSTEM,
    _annotate_diff_with_line_numbers,
)
from src.intelligence.passes.candidates import generate_candidates
from src.intelligence.passes.pipeline import FileReviewInput
from src.intelligence.retrieval import FileContext
from src.intelligence.schema import Finding

logger = logging.getLogger(__name__)

_MAX_FILE_LINES = 120

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_function",
            "description": (
                "Return the full source of a named function in a file changed in this PR."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repository-relative file path",
                    },
                    "name": {
                        "type": "string",
                        "description": "Function or method name",
                    },
                },
                "required": ["path", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file",
            "description": (
                "Return up to 120 lines of a file from this PR (read-only)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repository-relative file path",
                    },
                },
                "required": ["path"],
            },
        },
    },
]


def _execute_tool(
    name: str,
    arguments: dict[str, Any],
    path_to_content: dict[str, str],
    mod_funcs_by_path: dict[str, list[FunctionChunk]],
) -> str:
    path = (arguments.get("path") or "").strip()
    if not path or path not in path_to_content:
        return "[not found in PR — only changed files in this pull request are available]"

    if name == "get_file":
        lines = (path_to_content.get(path) or "").splitlines()
        if not lines:
            return "[file is empty or unavailable]"
        snippet = "\n".join(f"{i + 1:4d} | {ln}" for i, ln in enumerate(lines[:_MAX_FILE_LINES]))
        if len(lines) > _MAX_FILE_LINES:
            snippet += f"\n... ({len(lines) - _MAX_FILE_LINES} more lines truncated)"
        return snippet

    if name == "get_function":
        func_name = (arguments.get("name") or "").strip()
        if not func_name:
            return "[name is required]"
        for fc in mod_funcs_by_path.get(path) or []:
            if fc.name == func_name:
                return fc.text or "[empty function body]"
        content = path_to_content.get(path) or ""
        if content and func_name in content:
            return (
                f"[function `{func_name}` not isolated by AST; showing file head]\n"
                + "\n".join(content.splitlines()[:_MAX_FILE_LINES])
            )
        return f"[function `{func_name}` not found in {path}]"

    return f"[unknown tool: {name}]"


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _message_to_dict(msg: Any) -> dict[str, Any]:
    if isinstance(msg, dict):
        return msg
    out: dict[str, Any] = {"role": getattr(msg, "role", "assistant")}
    content = getattr(msg, "content", None)
    if content is not None:
        out["content"] = content
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            out["tool_calls"].append(
                {
                    "id": getattr(tc, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(fn, "name", "") if fn else "",
                        "arguments": getattr(fn, "arguments", "{}") if fn else "{}",
                    },
                }
            )
    return out


async def _call_llm_with_tools(messages: list[dict[str, Any]]) -> Any:
    kwargs: dict[str, Any] = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "api_base": config.LLM_API_BASE or None,
        "timeout": 120.0,
    }
    # LiteLLM maps upstream 'error' finish_reason to 'stop' with empty content and
    # no tool_calls; without this guard the loop silently treats that as "no
    # findings". Log the raw response, retry with backoff, then raise so the
    # caller falls back to a deterministic review.
    for attempt in range(3):
        try:
            response = await acompletion(**kwargs)
        except Exception as e:
            if attempt == 2:
                raise
            logger.warning(
                "[agentic] tool call errored (%s), retry %d/2", e, attempt + 1
            )
            await asyncio.sleep(2 ** attempt)
            continue
        choice = response.choices[0]
        finish = getattr(choice, "finish_reason", None)
        msg = getattr(choice, "message", None)
        content = (getattr(msg, "content", None) or "") if msg else ""
        tool_calls = getattr(msg, "tool_calls", None) if msg else None
        if finish == "error" or (not content.strip() and not tool_calls):
            logger.warning(
                "[agentic] step finish_reason=%s, empty=%s (attempt %d/3); raw head: %r",
                finish, not content.strip(), attempt + 1, str(response)[:500],
            )
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"agentic step failed: finish_reason={finish}")
        return response
    raise RuntimeError("agentic step failed: retries exhausted")


async def _call_llm_final(messages: list[dict[str, Any]]) -> str:
    kwargs: dict[str, Any] = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "api_base": config.LLM_API_BASE or None,
        "timeout": 120.0,
    }
    response = await acompletion(**kwargs)
    return (response.choices[0].message.content or "").strip()


async def _findings_from_raw(raw: str, path: str) -> list[Finding]:
    from src.intelligence.llm_client import parse_with_repair
    from src.intelligence.passes.candidates import (
        _infer_category_from_body,
        _infer_certainty_from_body,
        _infer_impact_from_body,
    )

    async def _recall(repair_prompt: str) -> str:
        return await _call_llm_final(
            [
                {"role": "system", "content": REVIEW_FILE_SYSTEM},
                {"role": "user", "content": repair_prompt},
            ]
        )

    comments = await parse_with_repair(raw, path, _recall)
    findings: list[Finding] = []
    for c in comments:
        findings.append(
            Finding(
                path=path,
                line=c["line"],
                title="",
                body=c["body"],
                impact=_infer_impact_from_body(c["body"]),
                certainty=_infer_certainty_from_body(c["body"]),
                category=_infer_category_from_body(c["body"]),
                origin="agentic",
                fix=None,
                post_inline=c.get("post_inline", True),
            )
        )
    return findings


async def agentic_review(
    file_input: FileReviewInput,
    plan: EffortPlan,
    cap: ModelCapability,
    path_to_content: dict[str, str],
    mod_funcs_by_path: Optional[dict[str, list[FunctionChunk]]] = None,
    retrieval_ctx: Optional[FileContext] = None,
) -> list[Finding]:
    """Run a bounded tool-calling loop; fall back to generate_candidates on failure."""
    _ = plan
    _ = cap
    path = file_input.path
    file_diff = file_input.file_diff
    pr_context = file_input.pr_context or {}
    mod_funcs_by_path = mod_funcs_by_path or {}

    extra = (retrieval_ctx.agentic_context_block() if retrieval_ctx else "").strip()
    annotated = _annotate_diff_with_line_numbers(file_diff, path)
    user_content = (
        f"File: {path}\n\n"
        "Review the diff below. You may call tools to inspect other PR files or "
        "function bodies before emitting findings as a JSON array.\n\n"
    )
    if extra:
        user_content += extra + "\n\n---\n\n"
    user_content += annotated

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": REVIEW_FILE_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    max_steps = config.SIFT_AGENTIC_MAX_STEPS

    try:
        for step in range(max_steps):
            response = await _call_llm_with_tools(messages)
            msg = response.choices[0].message
            msg_dict = _message_to_dict(msg)
            messages.append(msg_dict)

            tool_calls = msg_dict.get("tool_calls") or []
            if not tool_calls:
                content = msg_dict.get("content") or ""
                return await _findings_from_raw(content, path)

            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                args = _parse_tool_arguments(fn.get("arguments"))
                result = _execute_tool(
                    name, args, path_to_content, mod_funcs_by_path
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id") or "",
                        "content": result,
                    }
                )
            logger.debug("[agentic] %s: step %d tool call(s)", path, step + 1)

        messages.append(
            {
                "role": "user",
                "content": (
                    "Step limit reached. Respond with your final JSON array of findings "
                    "only (no more tool calls)."
                ),
            }
        )
        final = await _call_llm_final(messages)
        return await _findings_from_raw(final, path)

    except Exception as e:
        logger.warning(
            "[agentic] %s: loop failed (%s), falling back to deterministic review",
            path,
            e,
        )
        return await generate_candidates(file_diff, path, pr_context)
