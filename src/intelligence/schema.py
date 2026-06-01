"""Core data types for review findings."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Impact(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TRIVIAL = "trivial"


class Certainty(str, Enum):
    CONFIRMED = "confirmed"
    LIKELY = "likely"
    SPECULATIVE = "speculative"


CATEGORIES = frozenset({
    "correctness",
    "security",
    "perf",
    "resource",
    "design",
    "maintainability",
    "style",
})


@dataclass
class Finding:
    path: str
    line: int
    title: str
    body: str
    impact: Impact
    certainty: Certainty
    category: str
    origin: str
    fix: Optional[str] = None
    post_inline: bool = True

    def severity(self) -> str:
        """Derive the legacy 5-tier severity label from impact × certainty."""
        return derive_severity(self.impact, self.certainty, self.category)

    def to_comment_dict(self) -> dict:
        """Adapter: produce the dict shape review_engine expects."""
        return {
            "path": self.path,
            "line": self.line,
            "body": self.body,
            "post_inline": self.post_inline,
        }


def derive_severity(
    impact: Impact,
    certainty: Certainty,
    category: str = "",
) -> str:
    """Map impact × certainty → bug/security/warning/suggestion/informational."""
    if category == "security" and impact in (Impact.CRITICAL, Impact.HIGH):
        return "security"
    if impact == Impact.CRITICAL:
        return "bug"
    if impact == Impact.HIGH:
        return "bug" if certainty != Certainty.SPECULATIVE else "warning"
    if impact == Impact.MEDIUM:
        return "warning"
    if impact == Impact.LOW:
        return "suggestion"
    return "informational"


_OLD_SEVERITY_TO_IMPACT: dict[str, Impact] = {
    "bug": Impact.HIGH,
    "security": Impact.HIGH,
    "warning": Impact.MEDIUM,
    "suggestion": Impact.LOW,
    "informational": Impact.TRIVIAL,
}

_OLD_SEVERITY_TO_CATEGORY: dict[str, str] = {
    "bug": "correctness",
    "security": "security",
    "warning": "correctness",
    "suggestion": "maintainability",
    "informational": "maintainability",
}


def confidence_to_certainty(confidence: int) -> Certainty:
    if confidence >= 8:
        return Certainty.CONFIRMED
    if confidence >= 7:
        return Certainty.LIKELY
    return Certainty.SPECULATIVE


def from_legacy_item(item: dict, path: str, body: str) -> Finding:
    """Build a Finding from existing LLM JSON output and formatted body."""
    old_sev = (item.get("severity") or "suggestion").lower()
    try:
        confidence = int(item.get("confidence", 7))
    except (TypeError, ValueError):
        confidence = 7

    certainty = confidence_to_certainty(confidence)
    if old_sev == "informational":
        certainty = Certainty.SPECULATIVE

    return Finding(
        path=path,
        line=int(item["line"]),
        title=(item.get("title") or "").strip() or "Issue",
        body=body,
        impact=_OLD_SEVERITY_TO_IMPACT.get(old_sev, Impact.LOW),
        certainty=certainty,
        category=_OLD_SEVERITY_TO_CATEGORY.get(old_sev, "maintainability"),
        origin="llm",
        fix=(item.get("fix") or None),
        post_inline=True,
    )
