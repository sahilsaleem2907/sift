"""Format past inline-comment feedback for LLM prompts."""
from typing import Any, Dict, List

from src.intelligence.llm_client import _is_placeholder_issue_title

_SEV_DISPLAY = {
    "bug": "Bug",
    "security": "Security",
    "warning": "Warning",
    "suggestion": "Suggestion",
}

_MAX_TITLE_CHARS = 140


def format_labeled_comment_examples(rows: List[Dict[str, Any]]) -> str:
    """Build an LLM section: explicit severity + title + reaction counts per past inline comment."""
    if not rows:
        return ""
    lines: List[str] = [
        "Past inline review comments on this repository (each row: severity, title text, reaction signal; "
        "calibrate tone and focus from these, do not copy wording):",
    ]
    for r in rows:
        sev_key = (r.get("severity") or "suggestion").lower()
        label = _SEV_DISPLAY.get(sev_key, sev_key.title())
        raw_title = (r.get("title") or "").strip()
        if _is_placeholder_issue_title(raw_title):
            raw_title = "(no extractable title; re-sync reactions to refresh)"
        elif not raw_title:
            raw_title = "(no title)"
        raw_title = raw_title.replace("\n", " ")
        if len(raw_title) > _MAX_TITLE_CHARS:
            raw_title = raw_title[: _MAX_TITLE_CHARS - 3] + "..."
        pos = int(r.get("positive") or 0)
        neg = int(r.get("negative") or 0)
        net = int(r.get("net", pos - neg))
        net_s = f"+{net}" if net > 0 else str(net)
        lines.append(
            f"- **Severity:** {label} · **Title:** {raw_title} · **Reactions:** "
            f"{pos} positive, {neg} negative (net {net_s})"
        )
    return "\n".join(lines)
