"""Health check endpoint."""
import logging

from fastapi import APIRouter
from sqlalchemy import text

from src import config
from src.storage.database import _get_engine

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=dict)
def health() -> dict:
    """Return service health status; verify Postgres connectivity."""
    try:
        with _get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"
    ok = db_status == "connected"
    return {
        "status": "ok" if ok else "degraded",
        "database": db_status,
        "model": config.LLM_MODEL,
    }
