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
    """Receive GitHub webhook; verify signature and dispatch pull_request events."""
    payload = await request.body()
    sig = request.headers.get("X-Hub-Signature-256")
    if not _verify_signature(payload, sig):
        logger.warning("Webhook signature verification failed")
        return Response(status_code=401, content="Invalid signature")

    event = request.headers.get("X-GitHub-Event", "")
    if event != "pull_request":
        return Response(status_code=200, content="ignored")

    try:
        body = json.loads(payload.decode("utf-8"))
    except Exception as e:
        logger.warning("Webhook body is not JSON: %s", e)
        return Response(status_code=400, content="Invalid JSON")

    action = body.get("action")
    if action not in ("opened", "synchronize"):
        return Response(status_code=200, content="ignored")

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

    from src.core.review_engine import run_review

    background_tasks.add_task(run_review, owner, repo, pr_number, installation_id)
    logger.info("Queued review for %s PR #%s", repo_full_name, pr_number)
    return Response(status_code=202, content="accepted")
