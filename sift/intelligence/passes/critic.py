# SPDX-License-Identifier: AGPL-3.0-only
"""Pass 2: critic / verification pass."""
import asyncio
import json
import logging
import re
from typing import Any, Optional

from sift import config
from sift.intelligence.capability import ModelCapability
from sift.intelligence.effort import EffortPlan
from sift.intelligence.llm_client import (
    _call_llm,
    _extract_json_array,
    cache_control_block,
    ignores_tool_choice,
    is_anthropic_direct,
    litellm_kwargs,
)
from sift.intelligence.prompts import CRITIC_BATCHED_SYSTEM, CRITIC_FINDING_SYSTEM
from sift.intelligence.rate_limit import call_with_limits, endpoint_key
from sift.intelligence.schema import Certainty, Finding, Impact
from sift.observability.cost import record_response

logger = logging.getLogger(__name__)

_IMPACT_RANK = {
    Impact.CRITICAL: 0,
    Impact.HIGH: 1,
    Impact.MEDIUM: 2,
    Impact.LOW: 3,
    Impact.TRIVIAL: 4,
}


def _impact_rank(impact: Impact) -> int:
    return _IMPACT_RANK.get(impact, 9)


def rule_dedupe(findings: list[Finding]) -> list[Finding]:
    """Drop duplicate (path, line) keys, keeping the highest-impact finding.

    critic_exempt findings always survive deduplication — they are never replaced
    by a lower-confidence LLM finding on the same line.
    """
    seen: dict[tuple[str, int], Finding] = {}
    for f in findings:
        key = (f.path, f.line)
        existing = seen.get(key)
        if existing is None:
            seen[key] = f
        elif f.critic_exempt and not existing.critic_exempt:
            # Static-tool finding always wins over an LLM finding on the same line
            seen[key] = f
        elif not f.critic_exempt and existing.critic_exempt:
            pass  # keep existing static-tool finding
        elif _impact_rank(f.impact) < _impact_rank(existing.impact):
            seen[key] = f
    return list(seen.values())


def _parse_impact(value: Any, default: Impact) -> Impact:
    if not value:
        return default
    try:
        return Impact(str(value).lower())
    except ValueError:
        return default


def _parse_certainty(value: Any, default: Certainty) -> Certainty:
    if not value:
        return default
    try:
        return Certainty(str(value).lower())
    except ValueError:
        return default


def _apply_verdict(finding: Finding, verdict_obj: dict) -> Optional[Finding]:
    verdict = (verdict_obj.get("verdict") or "keep").lower()
    # Hard guard: security or high-impact findings can never be dropped by the critic —
    # only certainty can be downgraded. This prevents the self-negating loop where the
    # same model that generated a finding talks itself out of it on the second call.
    if verdict == "drop" and (
        finding.category == "security" or finding.impact == Impact.CRITICAL
    ):
        logger.debug(
            "[critic] DROP overridden to KEEP for security/critical finding line=%d", finding.line
        )
        verdict = "keep"
    if verdict == "drop":
        return None
    rated_impact = _parse_impact(verdict_obj.get("impact"), finding.impact)

    return Finding(
        path=finding.path,
        line=finding.line,
        title=finding.title,
        body=finding.body,
        impact=rated_impact,
        certainty=_parse_certainty(verdict_obj.get("certainty"), finding.certainty),
        category=finding.category,
        origin=finding.origin,
        fix=finding.fix,
        post_inline=finding.post_inline,
    )


# Concurrency findings are inherently non-deterministic (no single triggering input), so the
# generator prompt tends to self-downgrade them to SPECULATIVE the same way it does for security —
# text-matched here (there is no dedicated "concurrency" category) so the clamp below can catch it.
_CONCURRENCY_RE = re.compile(
    r"\b(race condition|data race|toctou|goroutine|thread-safe|thread safety|"
    r"unsynchronized|double-checked locking|concurrent(?:ly)? (?:map|access|read|write|"
    r"modif|mutat)|without (?:a |proper )?(?:lock|synchroniz|atomic)|"
    r"concurrent map (?:read|write)\b)",
    re.IGNORECASE,
)


def _looks_like_concurrency(f: Finding) -> bool:
    return bool(_CONCURRENCY_RE.search(f"{f.title} {f.body}"))


def _clamp_certainty_for_critic(findings: list[Finding]) -> list[Finding]:
    """Raise certainty to at least LIKELY for security and concurrency findings.

    The generator prompt can self-downgrade security findings to SPECULATIVE when
    it's uncertain. The critic then sees SPECULATIVE and is tempted to drop. This
    clamp breaks that self-defeating loop: the critic must make an affirmative
    factual case to drop, not just exploit a low certainty score. (Drop
    prevention for security/CRITICAL findings lives in _apply_verdict.)

    Concurrency findings (races, TOCTOU, unsynchronized shared state) get the same
    treatment: by nature they have no single deterministic triggering input, so the
    generator readily rates them SPECULATIVE even when the shared-state hazard is
    real. Text-matched via _looks_like_concurrency since there is no dedicated
    "concurrency" category (see prompts.py's concurrency carve-out).

    All other findings keep their true certainty: a HIGH-impact correctness
    finding at SPECULATIVE should stay speculative so it renders as WARNING
    rather than being silently promoted back toward BUG.
    """
    clamped = []
    for f in findings:
        if f.certainty == Certainty.SPECULATIVE and (
            f.category == "security" or _looks_like_concurrency(f)
        ):
            f = Finding(
                path=f.path,
                line=f.line,
                title=f.title,
                body=f.body,
                impact=f.impact,
                certainty=Certainty.LIKELY,
                category=f.category,
                origin=f.origin,
                fix=f.fix,
                post_inline=f.post_inline,
                critic_exempt=f.critic_exempt,
            )
        clamped.append(f)
    return clamped


_BADGE_RE = re.compile(
    r"!\[[^\]]+\]\(https://img\.shields\.io/badge/[^)]+\)\s*",
    re.IGNORECASE,
)
_TITLE_RE = re.compile(
    r"!\[[^\]]+\]\(https://img\.shields\.io/badge/[^)]+\)\s*([^\n]+)",
    re.IGNORECASE,
)


def _plain_body(body: str) -> str:
    """Strip badge image markdown from a comment body to get plain text."""
    return _BADGE_RE.sub("", body or "").strip()


def _title_from_body(body: str) -> str:
    """Extract the title that follows the badge on the first line of the body."""
    m = _TITLE_RE.search(body or "")
    if m:
        return m.group(1).strip()
    # fallback: first non-empty line after stripping badges
    plain = _plain_body(body)
    return plain.splitlines()[0].strip()[:80] if plain else "(no title)"


def _critic_llm_kwargs() -> dict[str, Any]:
    return {
        "model": config.SIFT_REVIEW_MODEL or config.LLM_MODEL,
        "api_base": config.SIFT_REVIEW_MODEL_BASE_URL or config.LLM_API_BASE or None,
        "api_key": config.SIFT_REVIEW_MODEL_KEY or None,
    }


_CRITIC_TOOL_MAX_STEPS = 3


async def _acompletion_with_tools(
    messages: list[dict[str, Any]],
    llm_kw: dict[str, Any],
    reasoning_effort: Optional[str] = None,
) -> Any:
    from litellm import acompletion
    from sift.intelligence.passes.agentic import TOOLS

    kwargs: dict[str, Any] = {
        **litellm_kwargs(llm_kw["model"], llm_kw.get("api_base"), llm_kw.get("api_key")),
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    if reasoning_effort:
        # No temperature is set here, so Anthropic's default (1.0) already satisfies the
        # thinking constraint; max_tokens must exceed the thinking budget (4096 for "high").
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs["max_tokens"] = 8192
    resp = await call_with_limits(endpoint_key(kwargs), lambda: acompletion(**kwargs))
    record_response(llm_kw["model"], resp)
    return resp


async def _critique_with_tools(
    system: str,
    user_content: str,
    llm_kw: dict[str, Any],
    repo_root: str,
    path_to_content: dict[str, str],
    mod_funcs: dict[str, Any],
    cache_prefix: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Bounded tool-calling critic loop; returns the final raw model text.

    Symmetric with the agentic generator: the critic can call the same fact-tools to
    AFFIRMATIVELY verify a claim before dropping it (burden of proof on the drop).

    ``cache_prefix``: shared user text (PR title + diff + verification context) reused across
    every per-finding call on this file. For direct Anthropic review models it becomes a cached
    block so those calls — and the steps within each call — reuse it at ~0.1x.
    """
    from sift.intelligence.passes.agentic import (
        TOOLS,
        _execute_tool,
        _message_to_dict,
        _parse_tool_arguments,
    )

    if cache_prefix is not None and is_anthropic_direct(llm_kw["model"]):
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": cache_control_block(system, "1h")},
            {
                "role": "user",
                "content": cache_control_block(cache_prefix, "1h")
                + [{"type": "text", "text": user_content}],
            },
        ]
    else:
        content = user_content if cache_prefix is None else f"{cache_prefix}{user_content}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
    for step in range(_CRITIC_TOOL_MAX_STEPS):
        resp = await _acompletion_with_tools(messages, llm_kw, reasoning_effort=reasoning_effort)
        msg = resp.choices[0].message
        md = _message_to_dict(msg)
        messages.append(md)
        tool_calls = md.get("tool_calls") or []
        if not tool_calls:
            logger.info("[critic] tool loop answered after %d step(s)", step + 1)
            return md.get("content") or ""
        logger.info(
            "[critic] tool loop step %d/%d: %s", step + 1, _CRITIC_TOOL_MAX_STEPS,
            [(tc.get("function") or {}).get("name") for tc in tool_calls],
        )
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            args = _parse_tool_arguments(fn.get("arguments"))
            result = _execute_tool(name, args, path_to_content, mod_funcs, repo_root)
            messages.append(
                {"role": "tool", "tool_call_id": tc.get("id") or "", "content": result}
            )
    # Step budget exhausted — demand a final verdict with no more tool calls.
    from litellm import acompletion

    messages.append({
        "role": "user",
        "content": "Step limit reached. Respond with your final JSON verdict object only (no tool calls).",
    })
    final_kwargs: dict[str, Any] = {
        **litellm_kwargs(llm_kw["model"], llm_kw.get("api_base"), llm_kw.get("api_key")),
        "messages": messages,
    }
    if not ignores_tool_choice(llm_kw["model"], llm_kw.get("api_base")):
        # messages still carries tool_calls/tool results, so tools= must stay declared (direct
        # Anthropic rejects tool history without it). tool_choice="none" forces the final
        # verdict as text now that the step budget is spent.
        final_kwargs["tools"] = TOOLS
        final_kwargs["tool_choice"] = "none"
    # Otherwise (Ollama) tool_choice is ignored, so no tools are sent at all — the only
    # reliable way to get the text verdict there.
    if reasoning_effort:
        final_kwargs["reasoning_effort"] = reasoning_effort
        final_kwargs["max_tokens"] = 8192
    resp = await call_with_limits(endpoint_key(final_kwargs), lambda: acompletion(**final_kwargs))
    record_response(llm_kw["model"], resp)
    logger.info("[critic] tool loop budget (%d) exhausted, final verdict requested", _CRITIC_TOOL_MAX_STEPS)
    return resp.choices[0].message.content or ""


def _verification_context(pr_context: Optional[dict[str, Any]]) -> str:
    """Bounded code + runtime context so the critic can verify a claim against real code.

    Reuses blocks the pipeline already builds (callee bodies, changed-function before/after,
    caller usage) plus the detected runtime target. Empty string when nothing is available —
    the critic then keeps its safe-default behaviour.
    """
    if not pr_context:
        return ""
    parts: list[str] = []
    rt = pr_context.get("runtime_target")
    if rt:
        parts.append(
            f"Target runtime: {rt} (authoritative — an API/method/param that exists in this "
            "version is NOT a bug)."
        )
    notes = pr_context.get("external_api_notes")
    if notes:
        parts.append(notes)
    sba = pr_context.get("semantic_before_after")
    if sba and str(sba).strip():
        parts.append("Changed functions (before/after):\n" + str(sba).strip())
    callee = pr_context.get("callee_signatures")
    if callee and str(callee).strip():
        parts.append("Callee definitions from other PR files:\n" + str(callee).strip())
    caller = pr_context.get("caller_context")
    if caller:
        from sift.intelligence.llm_client import _format_caller_context
        block = _format_caller_context(caller)
        if block:
            parts.append(block)
    if not parts:
        return ""
    return (
        "\n\nVerification context (check the finding against this real code; "
        "drop it ONLY if this code affirmatively disproves the claim):\n"
        + "\n\n".join(parts)
    )


async def critique_batched(
    findings: list[Finding],
    diff: str,
    pr_title: str,
    cap: ModelCapability,
    pr_context: Optional[dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
) -> list[Finding]:
    """One LLM call per file with all candidates."""
    _ = cap
    if not findings:
        return []

    # critic_exempt findings (auto-promoted static tool findings) bypass the critic
    exempt = [f for f in findings if f.critic_exempt]
    to_critique = _clamp_certainty_for_critic([f for f in findings if not f.critic_exempt])
    if not to_critique:
        return exempt

    items = "\n".join(
        f"[{i}] line={f.line} impact={f.impact.value} certainty={f.certainty.value} "
        f"category={f.category} origin={f.origin}\n"
        f"     title: {f.title or _title_from_body(f.body)}\n"
        f"     description: {_plain_body(f.body)[:400]}"
        for i, f in enumerate(to_critique)
    )
    user_content = (
        f"PR title: {pr_title}\n\nDiff:\n{diff}"
        f"{_verification_context(pr_context)}\n\nProposed findings:\n{items}"
    )

    raw = await _call_llm(
        CRITIC_BATCHED_SYSTEM, user_content, reasoning_effort=reasoning_effort,
        **_critic_llm_kwargs(),
    )
    verdicts = _extract_json_array(raw) or []
    verdict_map: dict[int, dict] = {}
    for v in verdicts:
        if isinstance(v, dict) and "index" in v:
            try:
                verdict_map[int(v["index"])] = v
            except (TypeError, ValueError):
                continue

    kept: list[Finding] = []
    for i, f in enumerate(to_critique):
        v = verdict_map.get(i)
        if v is None:
            kept.append(f)
            logger.debug("[critic] KEEP line=%d (no verdict, safe default)", f.line)
            continue
        updated = _apply_verdict(f, v)
        if updated is None:
            logger.debug(
                "[critic] DROP line=%d reason=%s",
                f.line,
                v.get("reason", ""),
            )
            continue
        kept.append(updated)
        logger.debug(
            "[critic] KEEP line=%d impact=%s certainty=%s",
            updated.line,
            updated.impact.value,
            updated.certainty.value,
        )

    logger.info(
        "[critic] batched (model=%s): %d in -> %d kept (%d dropped) + %d exempt",
        config.SIFT_REVIEW_MODEL or config.LLM_MODEL,
        len(to_critique),
        len(kept),
        len(to_critique) - len(kept),
        len(exempt),
    )
    return exempt + kept


def _extract_json_object(raw: str) -> Optional[dict]:
    """Parse a single JSON object from critic per-finding response."""
    text = (raw or "").strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string: Optional[str] = None
    i = start
    while i < len(text):
        c = text[i]
        if in_string:
            if c == "\\" and i + 1 < len(text):
                i += 2
                continue
            if c == in_string:
                in_string = None
            i += 1
            continue
        if c in ('"', "'"):
            in_string = c
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
        i += 1
    return None


async def critique_per_finding(
    findings: list[Finding],
    diff: str,
    pr_title: str,
    cap: ModelCapability,
    pr_context: Optional[dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
) -> list[Finding]:
    """One LLM call per finding (high effort)."""
    _ = cap
    if not findings:
        return []

    exempt = [f for f in findings if f.critic_exempt]
    to_critique = _clamp_certainty_for_critic([f for f in findings if not f.critic_exempt])
    if not to_critique:
        return exempt

    verify_ctx = _verification_context(pr_context)
    # Symmetric tools: let the critic call the same fact-tools to verify before dropping.
    repo_root = (pr_context or {}).get("_repo_root")
    path_to_content = (pr_context or {}).get("_path_to_content") or {}
    mod_funcs = (pr_context or {}).get("_mod_funcs_by_path") or {}
    use_tools = bool(repo_root) and cap.supports_function_calling
    if use_tools:
        logger.debug("[critic] per-finding tool loop enabled (repo_root available)")

    kept: list[Finding] = []
    llm_kw = _critic_llm_kwargs()
    for idx, f in enumerate(to_critique):
        if idx > 0 and config.SIFT_LLM_REQUEST_DELAY > 0:
            await asyncio.sleep(config.SIFT_LLM_REQUEST_DELAY)
        # Shared prefix (PR title + diff + verification context) is identical for every finding
        # on this file — cache it so the per-finding calls reuse it. The finding-specific part
        # is the volatile suffix.
        shared_prefix = f"PR title: {pr_title}\n\nDiff:\n{diff}{verify_ctx}\n\n"
        finding_part = (
            f"Proposed finding:\n"
            f"line={f.line} impact={f.impact.value} certainty={f.certainty.value} "
            f"category={f.category} origin={f.origin}\n"
            f"title: {f.title or _title_from_body(f.body)}\n"
            f"description: {_plain_body(f.body)[:400]}"
        )
        if use_tools and repo_root:
            try:
                raw = await _critique_with_tools(
                    CRITIC_FINDING_SYSTEM, finding_part, llm_kw,
                    repo_root, path_to_content, mod_funcs,
                    cache_prefix=shared_prefix,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as e:
                logger.warning("[critic] tool loop failed (%s); plain critic fallback", e)
                raw = await _call_llm(
                    CRITIC_FINDING_SYSTEM, finding_part, cache_user_prefix=shared_prefix,
                    reasoning_effort=reasoning_effort, **llm_kw
                )
        else:
            raw = await _call_llm(
                CRITIC_FINDING_SYSTEM, finding_part, cache_user_prefix=shared_prefix,
                reasoning_effort=reasoning_effort, **llm_kw
            )
        v = _extract_json_object(raw)
        if v is None:
            kept.append(f)
            logger.debug("[critic] KEEP line=%d (parse failed, safe default)", f.line)
            continue
        updated = _apply_verdict(f, v)
        if updated is None:
            logger.debug(
                "[critic] DROP line=%d reason=%s",
                f.line,
                (v or {}).get("reason", ""),
            )
            continue
        kept.append(updated)
        logger.debug("[critic] KEEP line=%d (per-finding)", updated.line)

    logger.info(
        "[critic] per-finding: %d in -> %d kept (%d dropped) + %d exempt",
        len(to_critique),
        len(kept),
        len(to_critique) - len(kept),
        len(exempt),
    )
    return exempt + kept


async def critique(
    findings: list[Finding],
    diff: str,
    pr_title: str,
    plan: EffortPlan,
    cap: ModelCapability,
    pr_context: Optional[dict[str, Any]] = None,
) -> list[Finding]:
    """Run critic pass at the granularity defined by the effort plan."""
    # Request reasoning on critic calls only when the *review* model supports it — the critic
    # calls SIFT_REVIEW_MODEL (falling back to primary), whose capability may differ from `cap`.
    from sift.intelligence.capability import review_capability

    reasoning_effort = (
        plan.reasoning_effort if review_capability().supports_reasoning else None
    )
    if plan.critic_per_finding:
        return await critique_per_finding(
            findings, diff, pr_title, cap, pr_context, reasoning_effort=reasoning_effort
        )
    return await critique_batched(
        findings, diff, pr_title, cap, pr_context, reasoning_effort=reasoning_effort
    )
