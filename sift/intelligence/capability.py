# SPDX-License-Identifier: AGPL-3.0-only
"""Model capability detection for adapting calls and context budgets."""
import json
import logging
from dataclasses import dataclass
from typing import Optional

import httpx
import litellm

from sift import config

logger = logging.getLogger(__name__)

_CACHE: dict[tuple[str, Optional[str], Optional[str]], "ModelCapability"] = {}

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
    "claude-sonnet-4",
    "claude-haiku-4",
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


# LiteLLM provider prefixes that may front an Ollama server; anything else with a known
# provider prefix (anthropic/, groq/, openrouter/, ...) is never probed as Ollama.
_OLLAMA_COMPATIBLE_PREFIXES = ("ollama_chat/", "ollama/", "openai/")
_OLLAMA_SHOW_CACHE: dict[tuple[str, str], Optional[dict]] = {}


def ollama_model_name(model: str) -> str:
    """Model name as Ollama knows it: strip a LiteLLM ollama/openai routing prefix only.

    Other slashes are kept — Ollama names can contain them (e.g. ``hf.co/org/model``).
    """
    for prefix in _OLLAMA_COMPATIBLE_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def _ollama_root(api_base: str) -> str:
    base = api_base.rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


def _may_be_ollama(model: str) -> bool:
    if model.startswith(_OLLAMA_COMPATIBLE_PREFIXES):
        return True
    head = model.split("/", 1)[0] if "/" in model else ""
    return head not in litellm.provider_list


def _ollama_show_uncached(root: str, name: str, api_key: Optional[str]) -> Optional[dict]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.post(f"{root}/api/show", json={"model": name}, headers=headers, timeout=5.0)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception as exc:
        logger.debug("[capability] /api/show probe failed for %s @ %s: %s", name, root, exc)
        return None
    caps = data.get("capabilities") if isinstance(data, dict) else None
    if not isinstance(caps, list):
        return None
    ctx = None
    for key, value in (data.get("model_info") or {}).items():
        if key.endswith("context_length") and isinstance(value, int):
            ctx = value
            break
    return {"capabilities": [str(c) for c in caps], "context_length": ctx}


def ollama_show(model: str, api_base: Optional[str], api_key: Optional[str] = None) -> Optional[dict]:
    """Ask an Ollama server what it knows about ``model`` (cached per base + model).

    Returns ``{"capabilities": [...], "context_length": int | None}`` when ``api_base`` is an
    Ollama server that serves the model, else None. A 200 carrying a ``capabilities`` list is
    what identifies the endpoint as Ollama; any other base (OpenRouter, Groq, ...) 404s here.
    Never raises.
    """
    if not api_base or not _may_be_ollama(model):
        return None
    key = (_ollama_root(api_base), ollama_model_name(model))
    if key not in _OLLAMA_SHOW_CACHE:
        _OLLAMA_SHOW_CACHE[key] = _ollama_show_uncached(key[0], key[1], api_key)
    return _OLLAMA_SHOW_CACHE[key]


def is_ollama_endpoint(model: str, api_base: Optional[str], api_key: Optional[str] = None) -> bool:
    return ollama_show(model, api_base, api_key) is not None


def _detect_reasoning(model: str) -> bool:
    m = model.lower()
    return any(s in m for s in _REASONING_MODEL_SUBSTRINGS)


def detect(
    model: str,
    override_raw: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ModelCapability:
    """Return capability for a model string. Cached per (model, override, api_base).

    When ``api_base`` is an Ollama server, Ollama's own ``/api/show`` answer is authoritative
    (tools / thinking / real context length) — LiteLLM's model map doesn't know Ollama model
    ids and would report no tool support and an 8k context.

    For a model litellm doesn't recognize under a leading ``openrouter/`` prefix
    (e.g. brand-new ``openrouter/deepseek/deepseek-v4-*``), fall back to the
    native-provider name (``deepseek/deepseek-v4-*``) for the capability lookup.
    The full ``model`` string is still what callers use for the actual API call;
    only the lookup name is stripped, and only when the full name is unknown — so
    models litellm knows *only* under their openrouter name are never regressed.
    """
    cache_key = (model, override_raw, api_base)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    override = _from_override(override_raw)
    if override is not None:
        _CACHE[cache_key] = override
        return override

    show = ollama_show(model, api_base, api_key)
    if show is not None:
        caps = show["capabilities"]
        cap = ModelCapability(
            context_window=show["context_length"] or _CONSERVATIVE_DEFAULTS["context_window"],
            max_output_tokens=_CONSERVATIVE_DEFAULTS["max_output_tokens"],
            supports_function_calling="tools" in caps,
            supports_reasoning="thinking" in caps,
        )
        _CACHE[cache_key] = cap
        logger.debug(
            "[capability] model=%s source=ollama /api/show caps=%s ctx=%d",
            model, caps, cap.context_window,
        )
        return cap

    lookup = model
    info = None
    try:
        info = litellm.get_model_info(model)
    except Exception:
        if model.startswith("openrouter/"):
            lookup = model[len("openrouter/"):]
            try:
                info = litellm.get_model_info(lookup)
            except Exception:
                info = None

    ctx = _CONSERVATIVE_DEFAULTS["context_window"]
    max_out = _CONSERVATIVE_DEFAULTS["max_output_tokens"]
    if info:
        ctx = int(info.get("max_input_tokens") or info.get("max_tokens") or ctx)
        max_out = int(info.get("max_output_tokens") or max_out)

    fn_calling = False
    try:
        fn_calling = bool(litellm.supports_function_calling(model=lookup))
    except Exception:
        pass
    reasoning = _detect_reasoning(model)
    try:
        reasoning = bool(litellm.supports_reasoning(model=model)) or reasoning
    except Exception:
        pass

    cap = ModelCapability(
        context_window=ctx,
        max_output_tokens=max_out,
        supports_function_calling=fn_calling,
        supports_reasoning=reasoning,
    )
    _CACHE[cache_key] = cap
    logger.debug(
        "[capability] model=%s lookup=%s ctx=%d max_out=%d fn_calling=%s reasoning=%s",
        model,
        lookup,
        cap.context_window,
        cap.max_output_tokens,
        cap.supports_function_calling,
        cap.supports_reasoning,
    )
    return cap


def primary_capability() -> ModelCapability:
    """Capability for the primary LLM_MODEL."""
    return detect(
        config.LLM_MODEL,
        config.SIFT_CAPABILITY_OVERRIDE,
        api_base=config.LLM_API_BASE,
        api_key=config.LLM_API_KEY,
    )


def review_capability() -> ModelCapability:
    """Capability for SIFT_REVIEW_MODEL (critic/holistic); falls back to primary."""
    return detect(
        config.SIFT_REVIEW_MODEL or config.LLM_MODEL,
        config.SIFT_REVIEW_CAPABILITY_OVERRIDE,
        api_base=config.SIFT_REVIEW_MODEL_BASE_URL or config.LLM_API_BASE,
        api_key=config.SIFT_REVIEW_MODEL_KEY or config.LLM_API_KEY,
    )
