"""Evaluate whether a PR should be blocked based on collected review findings."""
from typing import Any, Dict, List, Tuple

from src import config
from src.intelligence.llm_client import extract_comment_severity_and_title


def evaluate_block_policy(comments: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Return (should_block, description) from LLM inline comment severities."""
    block_severities = set(config.SIFT_BLOCK_ON_SEVERITIES)
    counts: Dict[str, int] = {}
    for c in comments:
        sev, _ = extract_comment_severity_and_title(c.get("body", ""))
        counts[sev] = counts.get(sev, 0) + 1

    blocking_total = sum(v for k, v in counts.items() if k in block_severities)

    if blocking_total >= config.SIFT_BLOCK_MIN_FINDINGS:
        parts = [
            f"{counts[s]} {s}"
            for s in ("bug", "security", "warning", "suggestion")
            if counts.get(s, 0) > 0 and s in block_severities
        ]
        return True, f"Sift found {', '.join(parts)} issue(s) — merge blocked"

    total = sum(counts.values())
    if total:
        return False, f"Sift review passed ({total} non-blocking finding(s))"
    return False, "Sift review passed — no issues found"
