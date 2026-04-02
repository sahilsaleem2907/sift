"""Feedback collection: command parsing and reaction sync."""
import re
import logging
from typing import Optional

from src.feedback.enums import FeedbackCommand, ReactionContent
from src.integrations.github_client import GitHubClient, get_installation_token
from src.storage.database import get_review_by_repo_pr, store_reaction_event_if_new

logger = logging.getLogger(__name__)

# /feedback <verb>; allow word chars and hyphen
_FEEDBACK_RE = re.compile(r"/feedback\s+([\w-]+)", re.IGNORECASE)

_ALIAS_HELPFUL = frozenset({"helpful", "good", "yes"})
_ALIAS_NOT_HELPFUL = frozenset({"not-helpful", "not_helpful", "bad", "no"})


def parse_feedback_command(body: str) -> Optional[FeedbackCommand]:
    """Parse comment body for /feedback <verb>. Returns FeedbackCommand or None. No side effects."""
    if not body or not body.strip():
        return None
    m = _FEEDBACK_RE.search(body.strip())
    if not m:
        return None
    verb = m.group(1).strip().lower().replace("-", "_")
    if verb in _ALIAS_HELPFUL:
        return FeedbackCommand.helpful
    if verb in _ALIAS_NOT_HELPFUL:
        return FeedbackCommand.not_helpful
    if verb == "not_helpful":
        return FeedbackCommand.not_helpful
    return None


def _normalize_reaction_content(content: str) -> Optional[str]:
    """Map GitHub reaction content to our ReactionContent value (string)."""
    c = (content or "").strip().lower()
    if not c:
        return None
    allowed = {e.value for e in ReactionContent}
    return c if c in allowed else None


async def sync_reactions_for_pr(owner: str, repo: str, pr_number: int, installation_id: int) -> None:
    """Fetch reactions for the summary comment and our inline comments on this PR; store new ones (dedup)."""
    repo_full = f"{owner}/{repo}"
    review_data = get_review_by_repo_pr(repo_full, pr_number)
    if not review_data or review_data[1] is None:
        logger.debug("No review with comment_id for %s PR #%s", repo_full, pr_number)
        return
    db_review_id, summary_comment_id = review_data
    try:
        token = await get_installation_token(installation_id)
        async with GitHubClient(installation_id, token=token) as github:
            # Summary is an issue comment created via POST /issues/{pr_number}/comments.
            reactions = await github.get_comment_reactions(owner, repo, summary_comment_id)
            for r in reactions:
                user = r.get("user") or {}
                actor = user.get("login") or ""
                content = _normalize_reaction_content((r.get("content") or ""))
                if not actor or not content:
                    continue
                stored = store_reaction_event_if_new(
                    repo=repo_full,
                    pr_number=pr_number,
                    comment_id=summary_comment_id,
                    actor=actor,
                    reaction_content=content,
                    review_id=db_review_id,
                    is_inline_comment=False,
                )
                if stored:
                    logger.info("Stored reaction %s by %s on %s PR #%s (summary)", content, actor, repo_full, pr_number)

            # Inline comments (Files changed):
            # Prefer filtering to Sift's bot login, but some installation tokens may be forbidden from GET /user (403).
            # In that case, fall back to unfiltered inline comments so reactions can still be captured.
            all_inline = await github.list_pull_request_review_comments(owner, repo, pr_number)
            try:
                app_login = await github.get_authenticated_user_login()
            except Exception as e:
                logger.warning(
                    "Inline reaction sync: GET /user failed (falling back to unfiltered inline comments): %s",
                    e,
                )
                app_login = None

            if app_login:
                our_inline_ids = [
                    c["id"]
                    for c in all_inline
                    if (c.get("user") or {}).get("login") == app_login
                ]
            else:
                our_inline_ids = [c["id"] for c in all_inline]
            for inline_comment_id in our_inline_ids:
                try:
                    inline_reactions = await github.get_review_comment_reactions(owner, repo, inline_comment_id)
                except Exception as e:
                    logger.debug("Failed to get reactions for inline comment %s: %s", inline_comment_id, e)
                    continue
                for r in inline_reactions:
                    user = r.get("user") or {}
                    actor = user.get("login") or ""
                    content = _normalize_reaction_content((r.get("content") or ""))
                    if not actor or not content:
                        continue
                    stored = store_reaction_event_if_new(
                        repo=repo_full,
                        pr_number=pr_number,
                        comment_id=inline_comment_id,
                        actor=actor,
                        reaction_content=content,
                        review_id=db_review_id,
                        is_inline_comment=True,
                    )
                    if stored:
                        logger.info("Stored reaction %s by %s on %s PR #%s (inline)", content, actor, repo_full, pr_number)
    except Exception as e:
        logger.warning("Reaction sync failed for %s PR #%s: %s", repo_full, pr_number, e)
