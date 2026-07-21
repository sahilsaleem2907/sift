"""HTTP endpoint for triggering a PR review (e.g. from GitHub Actions)."""
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sift import config
from sift.core.review_engine import run_review
from sift.integrations.registry import get_forge_builder

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
    """JSON body for POST /review.

    ``provider`` selects the forge (default 'github', preserving the original contract).
    Credentials are provider-specific: GitHub uses ``github_token``/``installation_id``;
    other providers (e.g. Bitbucket) use the generic ``token`` field.
    """

    provider: str = "github"
    owner: str
    repo: str
    pr_number: int
    before_sha: Optional[str] = None
    github_token: Optional[str] = None
    installation_id: Optional[int] = None
    token: Optional[str] = None


@router.post("/review")
async def review(
    body: ReviewRequestBody,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> dict:
    """Trigger a PR review (CI flow, e.g. GitHub Actions / Bitbucket Pipelines).

    Auth via Bearer token if SIFT_API_KEY is set. The forge is selected by ``provider``
    and its builder is resolved from the forge-builder registry, so this endpoint is
    provider-agnostic. Returns 202 Accepted."""
    _check_api_key(authorization)

    try:
        factory = get_forge_builder(body.provider)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        forge_builder = await factory(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    background_tasks.add_task(
        run_review,
        forge_builder,
        body.owner,
        body.repo,
        body.pr_number,
        before_sha=body.before_sha,
    )
    logger.info(
        "Queued review for %s/%s PR #%s (provider=%s)",
        body.owner,
        body.repo,
        body.pr_number,
        body.provider,
    )
    return JSONResponse(
        content={"status": "accepted", "message": "Review queued"},
        status_code=202,
    )
