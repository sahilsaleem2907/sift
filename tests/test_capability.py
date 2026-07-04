"""Tests for src.intelligence.capability."""
import json
from unittest import mock

from src.intelligence.capability import ModelCapability, _CACHE, detect


def setup_function():
    _CACHE.clear()


def test_conservative_fallback():
    with mock.patch("litellm.get_model_info", side_effect=Exception("unknown")):
        with mock.patch("litellm.supports_function_calling", side_effect=Exception("no")):
            cap = detect("unknown/self-hosted-model-xyz")
    assert cap.context_window == 8192
    assert cap.supports_function_calling is False


def test_override_wins():
    override = json.dumps({
        "context_window": 32768,
        "supports_function_calling": True,
        "supports_reasoning": False,
    })
    cap = detect("any/model", override)
    assert cap.context_window == 32768
    assert cap.supports_function_calling is True
    assert cap.supports_reasoning is False


def test_invalid_override_does_not_crash():
    with mock.patch("litellm.get_model_info", side_effect=Exception("x")):
        with mock.patch("litellm.supports_function_calling", side_effect=Exception("x")):
            cap = detect("fallback/model", "not-json")
    assert cap.context_window == 8192


def test_reasoning_detection():
    cap = detect("anthropic/claude-opus-4-20250514")
    assert cap.supports_reasoning is True


def test_caching():
    call_count = 0

    def fake_info(model):
        nonlocal call_count
        call_count += 1
        return {"max_input_tokens": 16000, "max_output_tokens": 4096}

    with mock.patch("litellm.get_model_info", side_effect=fake_info):
        with mock.patch("litellm.supports_function_calling", return_value=False):
            detect("cached/model")
            detect("cached/model")
    assert call_count == 1


def test_openrouter_strip_fallback_when_full_name_unknown():
    """openrouter/<vendor>/<model> unknown to litellm -> strip and resolve native name."""
    def info(model):
        if model == "openrouter/deepseek/deepseek-v4-flash":
            raise Exception("unknown in openrouter map")
        assert model == "deepseek/deepseek-v4-flash"
        return {"max_input_tokens": 1000000, "max_output_tokens": 8192}

    def fc(model):
        return model == "deepseek/deepseek-v4-flash"  # True only for the stripped name

    with mock.patch("litellm.get_model_info", side_effect=info):
        with mock.patch("litellm.supports_function_calling", side_effect=fc):
            cap = detect("openrouter/deepseek/deepseek-v4-flash")
    assert cap.supports_function_calling is True
    assert cap.context_window == 1000000


def test_openrouter_known_name_not_stripped():
    """When litellm knows the full openrouter name, do NOT strip (no regression)."""
    seen = {}

    def fc(model):
        seen["fc_model"] = model
        return True

    with mock.patch("litellm.get_model_info", return_value={"max_input_tokens": 128000}):
        with mock.patch("litellm.supports_function_calling", side_effect=fc):
            cap = detect("openrouter/deepseek/deepseek-chat")
    assert seen["fc_model"] == "openrouter/deepseek/deepseek-chat"  # full name preserved
    assert cap.supports_function_calling is True


def test_per_role_override_keyed_in_cache():
    """Same model string with different overrides must not collide in the cache."""
    override = json.dumps({"context_window": 32768, "supports_function_calling": True})
    with mock.patch("litellm.get_model_info", side_effect=Exception("unknown")):
        with mock.patch("litellm.supports_function_calling", return_value=False):
            cap_plain = detect("same/model", None)
            cap_over = detect("same/model", override)
    assert cap_plain.supports_function_calling is False   # detected path
    assert cap_over.supports_function_calling is True      # override path
    assert cap_over.context_window == 32768
