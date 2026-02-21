"""PR diff extraction for review."""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.integrations.github_client import GitHubClient

logger = logging.getLogger(__name__)

# Match "diff --git a/<path> b/<path>" to extract file path (use b-side = new file)
_DIFF_GIT_RE = re.compile(r"^diff --git a/.+? b/(.+?)\s*$", re.MULTILINE)


def split_diff_by_file(diff: str) -> List[Tuple[str, str]]:
    """Split a full PR diff into per-file chunks.

    Parses unified diff and returns [(path, file_diff), ...] for each file.
    Path is the repository-relative path (same for a/ and b/ in diff --git).
    """
    if not diff or not diff.strip():
        return []

    parts: List[Tuple[str, str]] = []
    current_path: Optional[str] = None
    current_chunk: List[str] = []

    for line in diff.splitlines(keepends=True):
        m = _DIFF_GIT_RE.match(line)
        if m:
            if current_path is not None and current_chunk:
                parts.append((current_path, "".join(current_chunk)))
            current_path = m.group(1).strip()
            current_chunk = [line]
        elif current_path is not None:
            current_chunk.append(line)

    if current_path is not None and current_chunk:
        parts.append((current_path, "".join(current_chunk)))

    return parts


async def get_diff_for_review(
    owner: str,
    repo: str,
    pr_number: int,
    github_client: GitHubClient,
    include_context: bool = True,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Fetch PR diff and optional title/body context via the GitHub client.

    Returns:
        (diff_text, pr_context or None). pr_context is {"title": str, "body": str} when include_context.
    """
    diff = await github_client.get_pr_diff(owner, repo, pr_number)
    logger.info("Fetched diff for %s/%s PR #%s (%d chars)", owner, repo, pr_number, len(diff))

    pr_context: Optional[Dict[str, Any]] = None
    if include_context:
        try:
            details = await github_client.get_pr_details(owner, repo, pr_number)
            pr_context = {"title": details.get("title") or "", "body": details.get("body") or ""}
        except Exception as e:
            logger.warning("Could not fetch PR details for context: %s", e)

    return diff, pr_context
