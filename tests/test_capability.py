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
    with mock.patch("src.intelligence.capability.config") as cfg:
        cfg.SIFT_CAPABILITY_OVERRIDE = override
        cap = detect("any/model")
    assert cap.context_window == 32768
    assert cap.supports_function_calling is True
    assert cap.supports_reasoning is False


def test_invalid_override_does_not_crash():
    with mock.patch("src.intelligence.capability.config") as cfg:
        cfg.SIFT_CAPABILITY_OVERRIDE = "not-json"
        with mock.patch("litellm.get_model_info", side_effect=Exception("x")):
            with mock.patch("litellm.supports_function_calling", side_effect=Exception("x")):
                cap = detect("fallback/model")
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
