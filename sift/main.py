# SPDX-License-Identifier: AGPL-3.0-only
"""Application entrypoint: build the Sift app and register startup hooks."""
import logging

from sift.api.app import build_app
from sift import config

logger = logging.getLogger(__name__)

app = build_app()


def _log_review_config() -> None:
    """Log resolved review models and effort plan at startup (no secrets)."""
    from sift.intelligence.capability import (
        is_ollama_endpoint,
        primary_capability,
        review_capability,
    )
    from sift.intelligence.effort import current_plan, resolve_effort

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
                "Sift pipeline: LLM critic uses the primary model "
                "(set SIFT_REVIEW_MODEL for a separate one)"
            )

    prim = primary_capability()
    logger.info(
        "Sift capability primary: ctx=%d fn_calling=%s reasoning=%s ollama=%s",
        prim.context_window,
        prim.supports_function_calling,
        prim.supports_reasoning,
        is_ollama_endpoint(config.LLM_MODEL, config.LLM_API_BASE, config.LLM_API_KEY),
    )
    if plan.enable_agentic and not prim.supports_function_calling:
        logger.warning(
            "Sift pipeline: agentic review is enabled but %s reports no tool calling — "
            "every file will use the one-shot review. Set SIFT_CAPABILITY_OVERRIDE "
            '(e.g. {"supports_function_calling": true}) if the model does support tools.',
            config.LLM_MODEL,
        )
    if dedicated_critic:
        rev = review_capability()
        logger.info(
            "Sift capability critic: ctx=%d fn_calling=%s reasoning=%s ollama=%s",
            rev.context_window,
            rev.supports_function_calling,
            rev.supports_reasoning,
            is_ollama_endpoint(
                critic_model,
                config.SIFT_REVIEW_MODEL_BASE_URL or config.LLM_API_BASE,
                config.SIFT_REVIEW_MODEL_KEY or config.LLM_API_KEY,
            ),
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
        from sift.storage.database import init_db
        init_db()
    except Exception as e:
        logger.warning("DB init skipped or failed: %s", e)
