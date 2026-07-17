"""Bounded agentic review loop with tool-calling (Phase 4, high effort)."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from litellm import acompletion

from sift import config
from sift.intelligence.ast.function_extract import FunctionChunk
from sift.intelligence.capability import ModelCapability
from sift.intelligence.effort import EffortPlan
from sift.intelligence.llm_client import (
    REVIEW_FILE_SYSTEM,
    _annotate_diff_with_line_numbers,
    _parse_review_file_response,
)
from sift.intelligence.prompts import TEST_FILE_APPENDIX
from sift.intelligence.passes.candidates import generate_candidates
from sift.intelligence.passes.pipeline import FileReviewInput
from sift.intelligence.retrieval import FileContext
from sift.intelligence.schema import Finding

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
    return await acompletion(**kwargs)


async def _call_llm_final(messages: list[dict[str, Any]]) -> str:
    kwargs: dict[str, Any] = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "api_base": config.LLM_API_BASE or None,
        "timeout": 120.0,
    }
    response = await acompletion(**kwargs)
    return (response.choices[0].message.content or "").strip()


def _findings_from_raw(raw: str, path: str) -> list[Finding]:
    from sift.intelligence.passes.candidates import finding_from_comment

    comments = _parse_review_file_response(raw, path)
    return [finding_from_comment(c, path, origin="agentic") for c in comments]


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

    system_prompt = REVIEW_FILE_SYSTEM
    if pr_context.get("is_test"):
        system_prompt = REVIEW_FILE_SYSTEM + TEST_FILE_APPENDIX
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
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
                return _findings_from_raw(content, path)

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
        return _findings_from_raw(final, path)

    except Exception as e:
        logger.warning(
            "[agentic] %s: loop failed (%s), falling back to deterministic review",
            path,
            e,
        )
        return await generate_candidates(file_diff, path, pr_context)
