"""Tests for src.intelligence.effort."""
import os
from unittest import mock

from sift.intelligence.effort import EffortLevel, plan_for, resolve_effort


def test_plan_for_low():
    p = plan_for(EffortLevel.LOW)
    assert p.run_critic is False
    assert p.run_holistic is False
    assert p.context_depth == 0
    assert p.enable_agentic is False


def test_plan_for_balanced():
    p = plan_for(EffortLevel.BALANCED)
    assert p.run_critic is True
    assert p.critic_per_finding is False
    assert p.run_holistic is True
    assert p.context_depth == 1


def test_plan_for_high():
    p = plan_for(EffortLevel.HIGH)
    assert p.critic_per_finding is True
    assert p.enable_agentic is True
    assert p.context_depth == 2


def test_resolve_effort_valid():
    with mock.patch.dict(os.environ, {"SIFT_REVIEW_EFFORT": "high"}):
        import importlib
        from sift import config
        importlib.reload(config)
        from sift.intelligence import effort as effort_mod
        importlib.reload(effort_mod)
        assert effort_mod.resolve_effort() == EffortLevel.HIGH


def test_resolve_effort_invalid_falls_back():
    with mock.patch("sift.intelligence.effort.config") as cfg:
        cfg.SIFT_REVIEW_EFFORT = "not-a-level"
        assert resolve_effort() == EffortLevel.BALANCED
