"""Application entrypoint: FastAPI app, logging, and route mounting."""
import logging

from fastapi import FastAPI

from src import config
from src.api.health import router as health_router
from src.api.webhooks import router as webhooks_router

config.setup_logging()
config.validate_required()

logger = logging.getLogger(__name__)

app = FastAPI(title="Sift")
app.include_router(health_router)
app.include_router(webhooks_router)


@app.on_event("startup")
def on_startup() -> None:
    """Optional: ensure DB tables exist."""
    try:
        from src.storage.database import init_db
        init_db()
    except Exception as e:
        logger.warning("DB init skipped or failed: %s", e)
