# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded agentic review loop with tool-calling (Phase 4, high effort)."""
from __future__ import annotations

import asyncio
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
    _call_llm,
    _extract_json_array,
    _items_to_comments,
    _parse_review_file_response,
    cache_control_block,
    ignores_tool_choice,
    is_anthropic_direct,
    litellm_kwargs,
    parse_with_repair,
)
from sift.observability.cost import record_response
from sift.intelligence.prompts import TEST_FILE_APPENDIX
from sift.intelligence.rate_limit import call_with_limits, endpoint_key
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
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read up to 160 lines of ANY file in the repository (not just PR "
                "files), optionally a line range. Use to inspect definitions, base "
                "classes, or callers that live in unchanged code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository-relative file path"},
                    "start_line": {"type": "integer", "description": "1-based start line (optional)"},
                    "end_line": {"type": "integer", "description": "1-based end line (optional)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_repo",
            "description": "Regex-search the whole repository (like git grep). Returns path:line: text matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "A regular expression"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_definition",
            "description": "Find where a symbol (function/class/type) is DEFINED across the repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "A bare identifier"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_callers",
            "description": "Find call/usage sites of a symbol across the repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "A bare identifier"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_signature",
            "description": (
                "Return the definition/signature line(s) of a symbol. Use to verify a "
                "method exists and which parameters/keywords it actually accepts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "A bare identifier"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mro",
            "description": (
                "For a class, report its base classes, each base's abstract methods, "
                "and which abstract methods the class fails to implement (resolved "
                "repo-wide). Use to check abstract-method completeness or isinstance "
                "type relationships."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File containing the class"},
                    "class_name": {"type": "string", "description": "The class name"},
                },
                "required": ["path", "class_name"],
            },
        },
    },
]

# Terminal tool for the deterministic final answer. Forcing tool_choice to this
# tool makes the model return schema-constrained JSON in `arguments` instead of
# free text, so the output is guaranteed parseable (no reasoning-tag stripping /
# bracket scanning). Mirrors the freeform findings item schema exactly, so only
# the transport changes — _items_to_comments applies the identical filtering.
EMIT_FINDINGS_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_findings",
        "description": (
            "Submit the final review findings for this file as structured data. "
            "Call this exactly once with all findings; output [] for findings if "
            "there are no real issues."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line": {
                                "type": "integer",
                                "description": "1-based line number from a [L<n>] marker in the diff",
                            },
                            "severity": {"type": "string"},
                            "title": {"type": "string"},
                            "body": {"type": "string"},
                            "fix": {"type": "string"},
                            "confidence": {
                                "type": "integer",
                                "description": "1-10; findings below 5 are dropped",
                            },
                        },
                        "required": ["line", "severity", "title", "body", "confidence"],
                    },
                },
            },
            "required": ["findings"],
        },
    },
}


def _execute_tool(
    name: str,
    arguments: dict[str, Any],
    path_to_content: dict[str, str],
    mod_funcs_by_path: dict[str, list[FunctionChunk]],
    repo_root: Optional[str] = None,
) -> str:
    # Repo-wide tools (backed by the git checkout). Degrade gracefully when the
    # checkout is unavailable (e.g. eval harness with no clone).
    if name in ("read_file", "search_repo", "find_definition", "find_callers", "get_signature", "get_mro"):
        if not repo_root:
            return "[repo-wide tools unavailable: no checkout for this review]"
        from sift.core import code_intel
        if name == "read_file":
            return code_intel.read_file(
                repo_root, (arguments.get("path") or "").strip(),
                arguments.get("start_line"), arguments.get("end_line"),
            )
        if name == "search_repo":
            return code_intel.search_repo(repo_root, (arguments.get("pattern") or "").strip())
        if name == "find_definition":
            return code_intel.find_definition(repo_root, (arguments.get("symbol") or "").strip())
        if name == "find_callers":
            return code_intel.find_callers(repo_root, (arguments.get("symbol") or "").strip())
        if name == "get_signature":
            return code_intel.get_signature(repo_root, (arguments.get("symbol") or "").strip())
        if name == "get_mro":
            return code_intel.get_mro(
                repo_root, (arguments.get("path") or "").strip(),
                (arguments.get("class_name") or "").strip(),
            )

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


def _primary_kwargs() -> dict[str, Any]:
    return litellm_kwargs(config.LLM_MODEL, config.LLM_API_BASE, config.LLM_API_KEY)


async def _call_llm_with_tools(messages: list[dict[str, Any]]) -> Any:
    kwargs: dict[str, Any] = {
        **_primary_kwargs(),
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    # LiteLLM maps upstream 'error' finish_reason to 'stop' with empty content and
    # no tool_calls; without this guard the loop silently treats that as "no
    # findings". Log the raw response, retry with backoff, then raise so the
    # caller falls back to a deterministic review. Exceptions are already retried
    # by call_with_limits, so they propagate straight to the caller.
    for attempt in range(3):
        response = await call_with_limits(endpoint_key(kwargs), lambda: acompletion(**kwargs))
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
        record_response(config.LLM_MODEL, response)
        return response
    raise RuntimeError("agentic step failed: retries exhausted")


_FINAL_TEXT_INSTRUCTION = (
    "Step limit reached. Respond now with your final findings as the JSON array described in "
    "your instructions (use [] if there are none). Do not call any tools."
)


async def _call_llm_final(messages: list[dict[str, Any]]) -> str:
    kwargs: dict[str, Any] = {
        **_primary_kwargs(),
        "messages": messages + [{"role": "user", "content": _FINAL_TEXT_INSTRUCTION}],
    }
    if not ignores_tool_choice(config.LLM_MODEL, config.LLM_API_BASE):
        # messages still carries the loop's tool_calls/tool results, so tools= must stay
        # declared (direct Anthropic rejects tool history without it). tool_choice="none"
        # forces a text answer instead of another tool call.
        kwargs["tools"] = TOOLS
        kwargs["tool_choice"] = "none"
    # Otherwise (Ollama) tool_choice is ignored, and the model keeps calling tools it saw in
    # history even when only emit_findings is declared — sending no tools at all is the only
    # reliable way to get a text answer there.
    response = await call_with_limits(endpoint_key(kwargs), lambda: acompletion(**kwargs))
    record_response(config.LLM_MODEL, response)
    return (response.choices[0].message.content or "").strip()


async def _emit_findings_final(messages: list[dict[str, Any]], path: str) -> list[dict[str, Any]]:
    """Ask the model to submit findings via the emit_findings tool.

    Deterministic: the response is schema-constrained JSON in the tool-call
    arguments, so there is no free text to heuristically parse. Forcing a specific
    tool is safe here only because the agentic loop sends no reasoning_effort —
    Anthropic forbids a forced tool_choice under extended thinking. If reasoning is
    ever enabled on this path, this call must be gated. Endpoints that ignore
    tool_choice (Ollama) may answer with another tool call instead; that raises so
    the caller falls through to the no-tools text path.
    """
    emit_messages = messages + [{
        "role": "user",
        "content": "Step limit reached. Submit your findings now by calling emit_findings.",
    }]
    kwargs: dict[str, Any] = {
        **_primary_kwargs(),
        "messages": emit_messages,
        # Prior read-tool calls remain in history, so their definitions must stay
        # declared alongside emit_findings (direct-Anthropic requirement).
        "tools": TOOLS + [EMIT_FINDINGS_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "emit_findings"}},
    }
    response = await call_with_limits(endpoint_key(kwargs), lambda: acompletion(**kwargs))
    record_response(config.LLM_MODEL, response)
    tool_calls = getattr(response.choices[0].message, "tool_calls", None) or []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        if fn and getattr(fn, "name", "") == "emit_findings":
            args = _parse_tool_arguments(getattr(fn, "arguments", "{}"))
            findings = args.get("findings")
            if isinstance(findings, list):
                return _items_to_comments(findings, path)
            return []
    returned = [getattr(getattr(tc, "function", None), "name", "?") for tc in tool_calls]
    raise RuntimeError(f"emit_findings not returned (got tool calls: {returned or 'none'})")


async def _final_comments(
    messages: list[dict[str, Any]], path: str, cap: ModelCapability
) -> list[dict[str, Any]]:
    """Produce the final findings as comment dicts, deterministically when possible.

    Tier 1: ask for the emit_findings tool (schema-constrained JSON, always parseable).
    Tier 2: for models that can't call tools, or if tier 1 doesn't return emit_findings,
    fall back to a free-text final call routed through parse_with_repair.
    """
    if cap.supports_function_calling:
        try:
            comments = await _emit_findings_final(messages, path)
            logger.info("[agentic] %s: emit_findings returned %d finding(s)", path, len(comments))
            return comments
        except Exception as e:
            logger.warning(
                "[agentic] %s: emit_findings failed (%s); free-text final answer",
                path, e,
            )
    raw = await _call_llm_final(messages)

    async def _recall(repair_prompt: str) -> str:
        return await _call_llm(REVIEW_FILE_SYSTEM, repair_prompt)

    comments = await parse_with_repair(raw, path, _recall)
    logger.info("[agentic] %s: free-text final answer gave %d finding(s)", path, len(comments))
    return comments


def _comments_to_findings(comments: list[dict[str, Any]], path: str) -> list[Finding]:
    from sift.intelligence.passes.candidates import finding_from_comment

    return [finding_from_comment(c, path, origin="agentic") for c in comments]


def _findings_from_raw(raw: str, path: str) -> list[Finding]:
    return _comments_to_findings(_parse_review_file_response(raw, path), path)


async def agentic_review(
    file_input: FileReviewInput,
    plan: EffortPlan,
    cap: ModelCapability,
    path_to_content: dict[str, str],
    mod_funcs_by_path: Optional[dict[str, list[FunctionChunk]]] = None,
    retrieval_ctx: Optional[FileContext] = None,
    repo_root: Optional[str] = None,
    context_window: Optional[int] = None,
) -> list[Finding]:
    """Run a bounded tool-calling loop; fall back to generate_candidates on failure.

    Tool results are capped, and if the provider rejects the growing history as too long the
    latest tool turns are dropped and the final answer is requested from what remains.
    """
    from sift.intelligence.budget import (
        cap_tool_result,
        drop_tool_turns,
        input_budget,
        overflow_shrink_factor,
    )

    _ = plan
    context_window = context_window or cap.context_window
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
    # Grounding the per-file reviewer already gets (review_file) but the agentic loop
    # previously omitted: the authoritative target runtime and curated footgun notes.
    runtime_target = pr_context.get("runtime_target")
    if runtime_target:
        user_content += (
            f"Target runtime for this file: {runtime_target} — AUTHORITATIVE. Assume every "
            "API/method/parameter available in this version exists; do not add 'older version' "
            "caveats.\n\n"
        )
    external_api_notes = pr_context.get("external_api_notes")
    if external_api_notes:
        user_content += external_api_notes + "\n\n"
    if extra:
        user_content += extra + "\n\n---\n\n"
    user_content += annotated

    system_prompt = REVIEW_FILE_SYSTEM
    if pr_context.get("is_test"):
        system_prompt = REVIEW_FILE_SYSTEM + TEST_FILE_APPENDIX
    # Cache the stable prefix (system + the initial file/diff/context user turn) so the
    # remaining tool-loop steps resend it at ~0.1x instead of full price — the dominant cost
    # of the high-effort primary pass. The system prompt alone is below Opus's 4096-token
    # cache minimum, so the breakpoint that matters is on the (large) initial user turn; it
    # caches everything before it. 5m TTL: this context is unique per file, reused only within
    # this loop, so the cheaper 5m write beats a 1h write. Non-Anthropic providers get plain
    # strings (unchanged).
    if is_anthropic_direct(config.LLM_MODEL):
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": cache_control_block(system_prompt, "1h")},
            {"role": "user", "content": cache_control_block(user_content, "5m")},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    max_steps = config.SIFT_AGENTIC_MAX_STEPS

    try:
        for step in range(max_steps):
            try:
                response = await _call_llm_with_tools(messages)
            except Exception as e:
                if overflow_shrink_factor(e) is None or len(messages) <= 2:
                    raise
                popped = drop_tool_turns(messages, input_budget(context_window, False))
                logger.warning(
                    "[agentic] %s: history too long at step %d; dropped %d tool turn(s), "
                    "requesting final answer", path, step + 1, popped,
                )
                break
            msg = response.choices[0].message
            msg_dict = _message_to_dict(msg)
            messages.append(msg_dict)

            tool_calls = msg_dict.get("tool_calls") or []
            if not tool_calls:
                # Model stopped calling tools and answered in text. Trust a clean
                # parse (incl. a legit empty []); only when NO array is extractable
                # at all (a parse failure) fall back to the deterministic emit call.
                content = msg_dict.get("content") or ""
                comments = _parse_review_file_response(content, path)
                if not comments and content.strip() and _extract_json_array(content) is None:
                    comments = await _final_comments(messages, path, cap)
                logger.info(
                    "[agentic] %s: answered after %d step(s) with %d finding(s)",
                    path, step + 1, len(comments),
                )
                return _comments_to_findings(comments, path)

            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                args = _parse_tool_arguments(fn.get("arguments"))
                result = _execute_tool(
                    name, args, path_to_content, mod_funcs_by_path, repo_root
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id") or "",
                        "content": cap_tool_result(result),
                    }
                )
            logger.info(
                "[agentic] %s: step %d/%d: %d tool call(s) %s",
                path, step + 1, max_steps, len(tool_calls),
                [(tc.get("function") or {}).get("name") for tc in tool_calls],
            )

        # Step budget exhausted (or history too long): get the final answer deterministically
        # (emit tool when supported; free-text + repair otherwise). Each tier appends its own
        # "submit now" instruction, so no extra prompt is needed here.
        logger.info("[agentic] %s: step loop ended, requesting final answer", path)
        comments = await _final_comments(messages, path, cap)
        return _comments_to_findings(comments, path)

    except Exception as e:
        logger.warning(
            "[agentic] %s: loop failed (%s), falling back to deterministic review",
            path,
            e,
        )
        return await generate_candidates(
            file_diff, path, pr_context, context_window=context_window
        )
