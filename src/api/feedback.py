"""HTTP endpoint for feedback sync (e.g. GitHub Actions on PR closed)."""
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.review import _check_api_key
from src.feedback.collector import sync_reactions_for_pr
from src.storage.database import store_pr_closed_event

router = APIRouter()
logger = logging.getLogger(__name__)


class FeedbackRequestBody(BaseModel):
    """JSON body for POST /feedback."""

    owner: str
    repo: str
    pr_number: int
    merged: bool
    github_token: Optional[str] = None
    installation_id: Optional[int] = None


@router.post("/feedback")
async def feedback(
    body: FeedbackRequestBody,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> dict:
    """Sync reactions into DB and record PR closed/merged (same as webhook on pull_request closed).

    Provide exactly one of ``github_token`` (e.g. ``GITHUB_TOKEN`` in Actions) or ``installation_id`` (App).
    """
    _check_api_key(authorization)

    has_token = bool(body.github_token)
    has_installation = body.installation_id is not None
    if has_token == has_installation:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of github_token or installation_id",
        )

    repo_full = f"{body.owner}/{body.repo}"
    try:
        store_pr_closed_event(repo_full, body.pr_number, body.merged)
    except Exception as e:
        logger.warning("Failed to store pr_closed event: %s", e)

    background_tasks.add_task(
        sync_reactions_for_pr,
        body.owner,
        body.repo,
        body.pr_number,
        body.installation_id,
        body.github_token,
    )
    logger.info(
        "Queued feedback sync for %s PR #%s (auth=%s)",
        repo_full,
        body.pr_number,
        "token" if has_token else "installation_id",
    )
    return JSONResponse(
        content={"status": "accepted", "message": "Feedback sync queued"},
        status_code=202,
    )
