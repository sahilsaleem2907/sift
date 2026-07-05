"""Pass 2: critic / verification pass."""
import asyncio
import json
import logging
import re
from typing import Any, Optional

from src import config
from src.intelligence.capability import ModelCapability
from src.intelligence.effort import EffortPlan
from src.intelligence.llm_client import _call_llm, _extract_json_array
from src.intelligence.prompts import CRITIC_BATCHED_SYSTEM, CRITIC_FINDING_SYSTEM
from src.intelligence.schema import Certainty, Finding, Impact

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


def _clamp_certainty_for_critic(findings: list[Finding]) -> list[Finding]:
    """Raise certainty to at least LIKELY for security/high-impact findings.

    The generator prompt can self-downgrade security findings to SPECULATIVE when
    it's uncertain. The critic then sees SPECULATIVE and is tempted to drop. This
    clamp breaks that self-defeating loop: the critic must make an affirmative
    factual case to drop, not just exploit a low certainty score.

    Only affects non-exempt findings passed to the critic; does not change the
    final Finding stored in output (certainty is re-rated by the critic anyway).
    """
    clamped = []
    for f in findings:
        if (
            f.certainty == Certainty.SPECULATIVE
            and (f.category == "security" or f.impact in (Impact.CRITICAL, Impact.HIGH))
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
    sba = pr_context.get("semantic_before_after")
    if sba and str(sba).strip():
        parts.append("Changed functions (before/after):\n" + str(sba).strip())
    callee = pr_context.get("callee_signatures")
    if callee and str(callee).strip():
        parts.append("Callee definitions from other PR files:\n" + str(callee).strip())
    caller = pr_context.get("caller_context")
    if caller:
        from src.intelligence.llm_client import _format_caller_context
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

    raw = await _call_llm(CRITIC_BATCHED_SYSTEM, user_content, **_critic_llm_kwargs())
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
    kept: list[Finding] = []
    llm_kw = _critic_llm_kwargs()
    for idx, f in enumerate(to_critique):
        if idx > 0 and config.SIFT_LLM_REQUEST_DELAY > 0:
            await asyncio.sleep(config.SIFT_LLM_REQUEST_DELAY)
        user_content = (
            f"PR title: {pr_title}\n\nDiff:\n{diff}{verify_ctx}\n\n"
            f"Proposed finding:\n"
            f"line={f.line} impact={f.impact.value} certainty={f.certainty.value} "
            f"category={f.category} origin={f.origin}\n"
            f"title: {f.title or _title_from_body(f.body)}\n"
            f"description: {_plain_body(f.body)[:400]}"
        )
        raw = await _call_llm(CRITIC_FINDING_SYSTEM, user_content, **llm_kw)
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
    if plan.critic_per_finding:
        return await critique_per_finding(findings, diff, pr_title, cap, pr_context)
    return await critique_batched(findings, diff, pr_title, cap, pr_context)
