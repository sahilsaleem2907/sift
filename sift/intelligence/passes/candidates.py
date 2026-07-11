"""Pass 1: per-file candidate generation.

Phase 1 behaviour: delegates directly to review_file() in llm_client.py and wraps
the output in Finding objects.
"""
import logging
import re
from typing import Any, Optional

from sift.intelligence import llm_client
from sift.intelligence.schema import Certainty, Finding, Impact

logger = logging.getLogger(__name__)

_SEV_BADGE_RE = re.compile(
    r"!\[(BUG|SECURITY|WARNING|SUGGESTION|INFORMATIONAL)\]",
    re.IGNORECASE,
)


_BADGE_TO_CATEGORY: dict[str, str] = {
    "bug": "correctness",
    "security": "security",
    "warning": "correctness",
    "suggestion": "maintainability",
    "informational": "maintainability",
}


def _infer_impact_from_body(body: str) -> Impact:
    m = _SEV_BADGE_RE.search(body)
    if m:
        sev = m.group(1).lower()
        if sev in ("bug", "security"):
            return Impact.HIGH
        if sev == "warning":
            return Impact.MEDIUM
        if sev == "suggestion":
            return Impact.LOW
        return Impact.LOW
    low = body.lower()
    if "bug" in low or "security" in low:
        return Impact.HIGH
    if "warning" in low:
        return Impact.MEDIUM
    return Impact.LOW


def _infer_category_from_body(body: str) -> str:
    m = _SEV_BADGE_RE.search(body)
    if m:
        return _BADGE_TO_CATEGORY.get(m.group(1).lower(), "correctness")
    return "correctness"


def _infer_certainty_from_body(body: str) -> Certainty:
    m = _SEV_BADGE_RE.search(body)
    if m and m.group(1).lower() == "informational":
        return Certainty.SPECULATIVE
    if "informational" in body.lower():
        return Certainty.SPECULATIVE
    return Certainty.LIKELY


async def generate_candidates(
    file_diff: str,
    path: str,
    pr_context: Optional[dict[str, Any]] = None,
) -> list[Finding]:
    """Call the LLM for a single file and return findings as Finding objects."""
    raw_comments = await llm_client.review_file(file_diff, path, pr_context)
    findings: list[Finding] = []
    for c in raw_comments:
        findings.append(
            Finding(
                path=path,
                line=c["line"],
                title="",
                body=c["body"],
                impact=_infer_impact_from_body(c["body"]),
                certainty=_infer_certainty_from_body(c["body"]),
                category=_infer_category_from_body(c["body"]),
                origin="llm",
                fix=None,
                post_inline=c.get("post_inline", True),
            )
        )
    return findings
