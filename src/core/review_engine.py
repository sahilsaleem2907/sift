"""Orchestrate PR review: diff -> LLM -> DB -> comment."""
import asyncio
import logging

from src.integrations.github_client import GitHubClient, get_installation_token
from src.core.pr_analyzer import get_diff_for_review
from src.intelligence.llm_client import review as llm_review
from src.storage.database import store_review

logger = logging.getLogger(__name__)


async def run_review(owner: str, repo: str, pr_number: int, installation_id: int) -> None:
    """Run the full review flow: fetch diff, get LLM review, persist, post comment.

    Logs and swallows exceptions so the webhook response is not affected.
    """
    repo_full = f"{owner}/{repo}"
    try:
        token = await get_installation_token(installation_id)
        async with GitHubClient(installation_id, token=token) as github:
            logger.info("Starting review for %s PR #%s", repo_full, pr_number)

            diff, pr_context = await get_diff_for_review(owner, repo, pr_number, github)
            if not diff.strip():
                logger.warning("Empty diff for %s PR #%s", repo_full, pr_number)
                return

            review_body = await llm_review(diff, pr_context)
            if not review_body.strip():
                logger.warning("Empty review from LLM for %s PR #%s", repo_full, pr_number)
                return

            comment_id = await github.create_comment(owner, repo, pr_number, review_body)
            try:
                store_review(repo_full, pr_number, installation_id, review_body, comment_id=comment_id)
            except Exception as e:
                logger.warning("Failed to store review in DB: %s", e)
            logger.info("Review completed for %s PR #%s", repo_full, pr_number)
    except Exception as e:
        logger.exception("Review failed for %s PR #%s: %s", repo_full, pr_number, e)
