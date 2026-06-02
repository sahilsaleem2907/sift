"""Application entrypoint: FastAPI app, logging, and route mounting."""
import logging

from fastapi import FastAPI

from src import config
from src.api.feedback import router as feedback_router
from src.api.health import router as health_router
from src.api.review import router as review_router
from src.api.webhooks import router as webhooks_router

config.setup_logging()
config.validate_required()

logger = logging.getLogger(__name__)

app = FastAPI(title="Sift")
app.include_router(health_router)
app.include_router(review_router)
app.include_router(feedback_router)
app.include_router(webhooks_router)


def _log_review_config() -> None:
    """Log resolved review models and effort plan at startup (no secrets)."""
    from src.intelligence.capability import primary_capability, review_capability
    from src.intelligence.effort import current_plan, resolve_effort

    effort = resolve_effort()
    plan = current_plan()
    critic_model = config.SIFT_REVIEW_MODEL or config.LLM_MODEL
    dedicated_critic = bool(config.SIFT_REVIEW_MODEL)

    logger.info(
        "Sift review: effort=%s context_depth=%d critic=%s holistic=%s "
        "agentic=%s agentic_max_steps=%d",
        effort.value,
        plan.context_depth,
        plan.run_critic,
        plan.run_holistic,
        plan.enable_agentic,
        config.SIFT_AGENTIC_MAX_STEPS,
    )
    logger.info(
        "Sift models: primary=%s api_base=%s",
        config.LLM_MODEL,
        config.LLM_API_BASE or "(provider default)",
    )
    if dedicated_critic:
        logger.info(
            "Sift models: critic/holistic=%s api_base=%s api_key_set=%s",
            critic_model,
            config.SIFT_REVIEW_MODEL_BASE_URL
            or config.LLM_API_BASE
            or "(provider default)",
            bool(config.SIFT_REVIEW_MODEL_KEY),
        )
    else:
        logger.info(
            "Sift models: critic/holistic=%s (same as primary)",
            critic_model,
        )
        if plan.run_critic:
            logger.info(
                "Sift pipeline: LLM critic uses rule_dedupe unless "
                "SIFT_REVIEW_MODEL is set to a separate model"
            )

    prim = primary_capability()
    logger.info(
        "Sift capability primary: ctx=%d fn_calling=%s reasoning=%s",
        prim.context_window,
        prim.supports_function_calling,
        prim.supports_reasoning,
    )
    if dedicated_critic:
        rev = review_capability()
        logger.info(
            "Sift capability critic: ctx=%d fn_calling=%s reasoning=%s",
            rev.context_window,
            rev.supports_function_calling,
            rev.supports_reasoning,
        )

    if config.VECTOR_DB_ENABLED:
        logger.info(
            "Sift embedding: model=%s api_base=%s",
            config.EMBEDDING_MODEL,
            config.EMBEDDING_API_BASE or "(provider default)",
        )

    if config.SIFT_CAPABILITY_OVERRIDE:
        logger.info("Sift capability: SIFT_CAPABILITY_OVERRIDE is set")

    if config.SIFT_REVIEW_MODEL and not config.SIFT_REVIEW_MODEL_KEY:
        logger.warning(
            "SIFT_REVIEW_MODEL is set but SIFT_REVIEW_MODEL_KEY is not; "
            "critic/holistic may fail for providers that require an explicit api_key"
        )


@app.on_event("startup")
def on_startup() -> None:
    """Log review config and ensure DB tables exist."""
    _log_review_config()
    try:
        from src.storage.database import init_db
        init_db()
    except Exception as e:
        logger.warning("DB init skipped or failed: %s", e)
