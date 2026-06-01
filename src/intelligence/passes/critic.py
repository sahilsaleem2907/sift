"""Pass 2: critic / verification pass."""
import asyncio
import json
import logging
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
    """Drop duplicate (path, line) keys, keeping the highest-impact finding."""
    seen: dict[tuple[str, int], Finding] = {}
    for f in findings:
        key = (f.path, f.line)
        if key not in seen or _impact_rank(f.impact) < _impact_rank(seen[key].impact):
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
    if (verdict_obj.get("verdict") or "keep").lower() == "drop":
        return None
    return Finding(
        path=finding.path,
        line=finding.line,
        title=finding.title,
        body=finding.body,
        impact=_parse_impact(verdict_obj.get("impact"), finding.impact),
        certainty=_parse_certainty(verdict_obj.get("certainty"), finding.certainty),
        category=finding.category,
        origin=finding.origin,
        fix=finding.fix,
        post_inline=finding.post_inline,
    )


def _critic_llm_kwargs() -> dict[str, Any]:
    return {
        "model": config.SIFT_REVIEW_MODEL or config.LLM_MODEL,
        "api_base": config.SIFT_REVIEW_MODEL_BASE_URL or config.LLM_API_BASE or None,
        "api_key": config.SIFT_REVIEW_MODEL_KEY or None,
    }


async def critique_batched(
    findings: list[Finding],
    diff: str,
    pr_title: str,
    cap: ModelCapability,
) -> list[Finding]:
    """One LLM call per file with all candidates."""
    _ = cap
    if not findings:
        return []

    items = "\n".join(
        f"[{i}] line={f.line} impact={f.impact.value} certainty={f.certainty.value} "
        f"category={f.category}\n"
        f"     title: {f.title or '(see body)'}\n"
        f"     body: {(f.body or '')[:300]}"
        for i, f in enumerate(findings)
    )
    user_content = f"PR title: {pr_title}\n\nDiff:\n{diff}\n\nProposed findings:\n{items}"

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
    for i, f in enumerate(findings):
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
        "[critic] batched: %d in -> %d kept (%d dropped)",
        len(findings),
        len(kept),
        len(findings) - len(kept),
    )
    return kept


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
) -> list[Finding]:
    """One LLM call per finding (high effort)."""
    _ = cap
    if not findings:
        return []

    kept: list[Finding] = []
    llm_kw = _critic_llm_kwargs()
    for idx, f in enumerate(findings):
        if idx > 0 and config.SIFT_LLM_REQUEST_DELAY > 0:
            await asyncio.sleep(config.SIFT_LLM_REQUEST_DELAY)
        user_content = (
            f"PR title: {pr_title}\n\nDiff:\n{diff}\n\n"
            f"Proposed finding:\n"
            f"line={f.line} impact={f.impact.value} certainty={f.certainty.value}\n"
            f"title: {f.title or '(see body)'}\n"
            f"body: {f.body}"
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
        "[critic] per-finding: %d in -> %d kept (%d dropped)",
        len(findings),
        len(kept),
        len(findings) - len(kept),
    )
    return kept


async def critique(
    findings: list[Finding],
    diff: str,
    pr_title: str,
    plan: EffortPlan,
    cap: ModelCapability,
) -> list[Finding]:
    """Run critic pass at the granularity defined by the effort plan."""
    if plan.critic_per_finding:
        return await critique_per_finding(findings, diff, pr_title, cap)
    return await critique_batched(findings, diff, pr_title, cap)
