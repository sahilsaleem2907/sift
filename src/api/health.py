"""Health check endpoint."""
import logging

from fastapi import APIRouter

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=dict)
def health() -> dict:
    """Return service health status."""
    return {"status": "ok"}
