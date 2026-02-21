"""PR diff extraction for review."""
import logging
from typing import Any, Dict, Optional, Tuple

from src.integrations.github_client import GitHubClient

logger = logging.getLogger(__name__)


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
