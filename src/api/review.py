"""HTTP endpoint for triggering a PR review (e.g. from GitHub Actions)."""
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src import config
from src.core.review_engine import run_review

router = APIRouter()
logger = logging.getLogger(__name__)


def _check_api_key(authorization: Optional[str] = None) -> None:
    """Raise 401 if SIFT_API_KEY is set and request does not bear it."""
    if not config.SIFT_API_KEY:
        return
    if not authorization or not authorization.strip().lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(maxsplit=1)[1].strip()
    if not token or token != config.SIFT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class ReviewRequestBody(BaseModel):
    """JSON body for POST /review."""

    owner: str
    repo: str
    pr_number: int
    before_sha: Optional[str] = None
    github_token: Optional[str] = None
    installation_id: Optional[int] = None


@router.post("/review")
async def review(
    body: ReviewRequestBody,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> dict:
    """Trigger a PR review (GitHub Actions flow). Auth via Bearer token if SIFT_API_KEY is set.
    Provide exactly one of github_token or installation_id. Returns 202 Accepted."""
    _check_api_key(authorization)

    has_token = bool(body.github_token)
    has_installation = body.installation_id is not None
    if has_token == has_installation:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of github_token or installation_id",
        )

    background_tasks.add_task(
        run_review,
        body.owner,
        body.repo,
        body.pr_number,
        installation_id=body.installation_id,
        github_token=body.github_token,
        before_sha=body.before_sha,
    )
    logger.info(
        "Queued review for %s/%s PR #%s (auth=%s)",
        body.owner,
        body.repo,
        body.pr_number,
        "token" if has_token else "installation_id",
    )
    return JSONResponse(
        content={"status": "accepted", "message": "Review queued"},
        status_code=202,
    )
