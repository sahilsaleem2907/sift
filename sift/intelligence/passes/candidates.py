"""Pass 1: per-file candidate generation.

Phase 1 behaviour: delegates directly to review_file() in llm_client.py and wraps
the output in Finding objects.
"""
import logging
from typing import Any, Optional

from sift.intelligence import llm_client
from sift.intelligence.schema import Certainty, Finding, Impact, from_legacy_item

logger = logging.getLogger(__name__)


def finding_from_comment(c: dict[str, Any], path: str, origin: str = "llm") -> Finding:
    """Build a Finding from a parsed review comment dict.

    Structured parses carry the LLM's severity label and confidence, which map to
    impact × certainty via from_legacy_item: bug conf 8-10 → HIGH+CONFIRMED (blocks),
    conf 7 → HIGH+LIKELY (blocks), conf 5-6 → HIGH+SPECULATIVE (renders as WARNING,
    non-blocking). Freeform parses have severity=None and get a LOW+LIKELY default.
    """
    if c.get("severity"):
        return from_legacy_item(
            {
                "line": c["line"],
                "severity": c["severity"],
                "title": c.get("title") or "",
                "confidence": c.get("confidence", 7),
                "fix": c.get("fix"),
            },
            path,
            c["body"],
            origin=origin,
            post_inline=c.get("post_inline", True),
        )
    return Finding(
        path=path,
        line=c["line"],
        title=(c.get("title") or "").strip(),
        body=c["body"],
        impact=Impact.LOW,
        certainty=Certainty.LIKELY,
        category="correctness",
        origin=origin,
        fix=None,
        post_inline=c.get("post_inline", True),
    )


async def generate_candidates(
    file_diff: str,
    path: str,
    pr_context: Optional[dict[str, Any]] = None,
) -> list[Finding]:
    """Call the LLM for a single file and return findings as Finding objects."""
    raw_comments = await llm_client.review_file(file_diff, path, pr_context)
    return [finding_from_comment(c, path) for c in raw_comments]
