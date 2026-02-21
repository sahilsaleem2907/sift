"""GitHub webhook handler."""
import hmac
import json
import logging
from hashlib import sha256
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Request, Response

from src import config

router = APIRouter()
logger = logging.getLogger(__name__)


def _verify_signature(payload: bytes, signature_header: Optional[str]) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    _, signature = signature_header.split("=", 1)
    secret = (config.GITHUB_WEBHOOK_SECRET or "").encode("utf-8")
    expected = hmac.new(secret, msg=payload, digestmod=sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks) -> Response:
    """Receive GitHub webhook; verify signature; dispatch pull_request or issue_comment."""
    payload = await request.body()
    sig = request.headers.get("X-Hub-Signature-256")
    if not _verify_signature(payload, sig):
        logger.warning("Webhook signature verification failed")
        return Response(status_code=401, content="Invalid signature")

    event = request.headers.get("X-GitHub-Event", "")
    try:
        body = json.loads(payload.decode("utf-8"))
    except Exception as e:
        logger.warning("Webhook body is not JSON: %s", e)
        return Response(status_code=400, content="Invalid JSON")
    
    if event == "pull_request":
        return _handle_pull_request(body, background_tasks)
    if event == "issue_comment":
        return _handle_issue_comment(body, background_tasks)
    return Response(status_code=200, content="ignored")


def _handle_pull_request(body: dict, background_tasks: BackgroundTasks) -> Response:
    """Queue review for opened/synchronize; sync reactions and record closed for closed."""
    action = body.get("action")
    repo_full_name = body.get("repository", {}).get("full_name")
    pr = body.get("pull_request", {})
    pr_number = pr.get("number")
    installation = body.get("installation") or {}
    installation_id = installation.get("id")

    if not repo_full_name or pr_number is None or not installation_id:
        logger.warning("Webhook missing repo, pr number, or installation id: %s", body)
        return Response(status_code=200, content="ignored")

    parts = repo_full_name.split("/", 1)
    owner = parts[0] if len(parts) == 2 else ""
    repo = parts[1] if len(parts) == 2 else repo_full_name
    if not owner or not repo:
        return Response(status_code=200, content="ignored")

    if action in ("opened", "synchronize"):
        from src.core.review_engine import run_review
        background_tasks.add_task(run_review, owner, repo, pr_number, installation_id)
        logger.info("Queued review for %s PR #%s", repo_full_name, pr_number)
        return Response(status_code=202, content="accepted")

    if action == "closed":
        from src.feedback.collector import sync_reactions_for_pr
        from src.storage.database import store_pr_closed_event
        background_tasks.add_task(sync_reactions_for_pr, owner, repo, pr_number, installation_id)
        merged = pr.get("merged", False)
        try:
            store_pr_closed_event(repo_full_name, pr_number, merged)
        except Exception as e:
            logger.warning("Failed to store pr_closed event: %s", e)
        logger.info("PR closed/merged: %s PR #%s merged=%s", repo_full_name, pr_number, merged)
        return Response(status_code=200, content="ok")

    return Response(status_code=200, content="ignored")


def _handle_issue_comment(body: dict, background_tasks: BackgroundTasks) -> Response:
    """Parse /feedback command, store event, and sync reactions for issue_comment created."""
    action = body.get("action")
    if action not in ("created", "edited"):
        return Response(status_code=200, content="ignored")

    repo_full_name = body.get("repository", {}).get("full_name")
    issue = body.get("issue", {})
    pr_number = issue.get("number")
    comment = body.get("comment", {})
    comment_id = comment.get("id")
    comment_body = comment.get("body") or ""
    comment_user = comment.get("user") or {}
    actor = comment_user.get("login") or ""
    installation = body.get("installation") or {}
    installation_id = installation.get("id")

    if not repo_full_name or pr_number is None or not installation_id:
        return Response(status_code=200, content="ignored")

    parts = repo_full_name.split("/", 1)
    owner = parts[0] if len(parts) == 2 else ""
    repo = parts[1] if len(parts) == 2 else repo_full_name
    if not owner or not repo:
        return Response(status_code=200, content="ignored")

    from src.feedback.collector import parse_feedback_command, sync_reactions_for_pr
    from src.feedback.enums import FeedbackEventType, FeedbackSource
    from src.storage.database import store_feedback_event

    cmd = parse_feedback_command(comment_body)
    if cmd is not None:
        store_feedback_event(
            event_type=FeedbackEventType.command.value,
            repo=repo_full_name,
            pr_number=pr_number,
            actor=actor,
            source=FeedbackSource.webhook.value,
            comment_id=comment_id,
            review_id=None,
            reaction_content=None,
            command=cmd.value,
        )
        logger.info("Stored /feedback %s from %s on %s PR #%s", cmd.value, actor, repo_full_name, pr_number)

    background_tasks.add_task(sync_reactions_for_pr, owner, repo, pr_number, installation_id)
    return Response(status_code=200, content="ok")
