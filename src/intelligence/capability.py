"""Model capability detection for adapting calls and context budgets."""
import json
import logging
from dataclasses import dataclass
from typing import Optional

import litellm

from src import config

logger = logging.getLogger(__name__)

_CACHE: dict[str, "ModelCapability"] = {}

_CONSERVATIVE_DEFAULTS = dict(
    context_window=8192,
    max_output_tokens=2048,
    supports_function_calling=False,
    supports_reasoning=False,
)

_REASONING_MODEL_SUBSTRINGS = (
    "o1",
    "o3",
    "thinking",
    "reasoning",
    "r1",
    "claude-opus-4",
    "claude-3-7",
)


@dataclass(frozen=True)
class ModelCapability:
    context_window: int
    max_output_tokens: int
    supports_function_calling: bool
    supports_reasoning: bool


def _from_override(raw: Optional[str]) -> Optional[ModelCapability]:
    if not raw:
        return None
    try:
        d = json.loads(raw)
        return ModelCapability(
            context_window=int(
                d.get("context_window", _CONSERVATIVE_DEFAULTS["context_window"])
            ),
            max_output_tokens=int(
                d.get("max_output_tokens", _CONSERVATIVE_DEFAULTS["max_output_tokens"])
            ),
            supports_function_calling=bool(d.get("supports_function_calling", False)),
            supports_reasoning=bool(d.get("supports_reasoning", False)),
        )
    except Exception as exc:
        logger.warning("SIFT_CAPABILITY_OVERRIDE is not valid JSON (%s); ignoring.", exc)
        return None


def _detect_reasoning(model: str) -> bool:
    m = model.lower()
    return any(s in m for s in _REASONING_MODEL_SUBSTRINGS)


def detect(model: str) -> ModelCapability:
    """Return capability for a model string. Results are cached per model string."""
    if model in _CACHE:
        return _CACHE[model]

    override = _from_override(config.SIFT_CAPABILITY_OVERRIDE)
    if override is not None:
        _CACHE[model] = override
        return override

    ctx = _CONSERVATIVE_DEFAULTS["context_window"]
    max_out = _CONSERVATIVE_DEFAULTS["max_output_tokens"]
    fn_calling = False
    try:
        info = litellm.get_model_info(model)
        ctx = int(info.get("max_input_tokens") or info.get("max_tokens") or ctx)
        max_out = int(info.get("max_output_tokens") or max_out)
    except Exception:
        pass
    try:
        fn_calling = bool(litellm.supports_function_calling(model=model))
    except Exception:
        pass

    cap = ModelCapability(
        context_window=ctx,
        max_output_tokens=max_out,
        supports_function_calling=fn_calling,
        supports_reasoning=_detect_reasoning(model),
    )
    _CACHE[model] = cap
    logger.debug(
        "[capability] model=%s ctx=%d max_out=%d fn_calling=%s reasoning=%s",
        model,
        cap.context_window,
        cap.max_output_tokens,
        cap.supports_function_calling,
        cap.supports_reasoning,
    )
    return cap


def primary_capability() -> ModelCapability:
    """Capability for the primary LLM_MODEL."""
    return detect(config.LLM_MODEL)


def review_capability() -> ModelCapability:
    """Capability for SIFT_REVIEW_MODEL (critic/holistic); falls back to primary."""
    return detect(config.SIFT_REVIEW_MODEL or config.LLM_MODEL)
