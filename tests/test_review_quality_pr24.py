"""Regression tests for the PR #24 severity-calibration fixes.

The reported failure mode: speculative error-handling findings ("missing
try/catch", "context exit may abort shutdown") rendered as BUG and blocked
merges. Three fixes cooperate here:
- the LLM's confidence now maps to certainty (bug + conf 5-6 → HIGH+SPECULATIVE),
- apply_final_severity_labels re-renders the badge from impact × certainty
  after the critic, so re-ratings reach the block policy,
- the prompt rubric demotes robustness concerns to warning/suggestion.
"""
import json

import pytest

from sift import config
from sift.core.block_policy import evaluate_block_policy
from sift.intelligence.effort import EffortLevel, plan_for
from sift.intelligence.llm_client import _parse_review_file_response
from sift.intelligence.passes.candidates import finding_from_comment
from sift.intelligence.passes.critic import _apply_verdict
from sift.intelligence.passes.severity import (
    apply_final_severity_labels,
    apply_severity_gate,
)
from sift.intelligence.prompts import (
    CRITIC_BATCHED_SYSTEM,
    CRITIC_FINDING_SYSTEM,
    REVIEW_FILE_SYSTEM,
)
from sift.intelligence.schema import Certainty, Impact


@pytest.fixture(autouse=True)
def _default_block_config(monkeypatch):
    monkeypatch.setattr(config, "SIFT_BLOCK_ON_SEVERITIES", ["bug", "security"])
    monkeypatch.setattr(config, "SIFT_BLOCK_MIN_FINDINGS", 1)


def _pipeline_comment(item: dict) -> dict:
    """Run one raw LLM item through parse → Finding → gate → final labels."""
    raw = json.dumps([item])
    comments = _parse_review_file_response(raw, "app/shutdown.py")
    assert len(comments) == 1
    finding = finding_from_comment(comments[0], "app/shutdown.py")
    plan = plan_for(EffortLevel.BALANCED)
    labeled = apply_final_severity_labels(apply_severity_gate([finding], plan))
    assert len(labeled) == 1
    return {"path": labeled[0].path, "line": labeled[0].line, "body": labeled[0].body}


def test_speculative_missing_try_catch_is_warning_and_does_not_block():
    comment = _pipeline_comment({
        "line": 42,
        "severity": "bug",
        "title": "Missing try/catch around IO",
        "body": "Context exit may abort shutdown if the write fails.",
        "confidence": 6,
    })
    assert "![WARNING]" in comment["body"]
    assert "![BUG]" not in comment["body"]
    should_block, _ = evaluate_block_policy([comment])
    assert should_block is False


def test_confirmed_bug_still_blocks():
    comment = _pipeline_comment({
        "line": 42,
        "severity": "bug",
        "title": "KeyError on deleted key",
        "body": "d[k] is accessed after k was deleted above; raises KeyError.",
        "confidence": 9,
    })
    assert "![BUG]" in comment["body"]
    should_block, _ = evaluate_block_policy([comment])
    assert should_block is True


def test_critic_downgrade_reaches_badge_and_block_policy():
    """A candidate the critic re-rates to speculative must stop blocking."""
    comments = _parse_review_file_response(json.dumps([{
        "line": 7,
        "severity": "bug",
        "title": "Unhandled exception in handler",
        "body": "The handler might raise on malformed input.",
        "confidence": 7,  # HIGH + LIKELY: would block if the critic said nothing
    }]), "app/handler.py")
    finding = finding_from_comment(comments[0], "app/handler.py")
    assert finding.impact == Impact.HIGH and finding.certainty == Certainty.LIKELY

    downgraded = _apply_verdict(finding, {
        "verdict": "keep",
        "impact": "high",
        "certainty": "speculative",
        "reason": "cannot confirm from visible diff",
    })
    plan = plan_for(EffortLevel.BALANCED)
    labeled = apply_final_severity_labels(apply_severity_gate([downgraded], plan))
    assert "![WARNING]" in labeled[0].body
    comment = {"path": labeled[0].path, "line": labeled[0].line, "body": labeled[0].body}
    should_block, _ = evaluate_block_policy([comment])
    assert should_block is False


# ---- prompt rubric content ----

def test_rubric_requires_what_when_consequence():
    for token in ("WHAT fails", "WHEN it fails", "CONSEQUENCE"):
        assert token in REVIEW_FILE_SYSTEM, f"{token!r} missing from severity rubric"


def test_rubric_maps_missing_try_catch_to_suggestion():
    lower = REVIEW_FILE_SYSTEM.lower()
    assert "missing try/catch" in lower
    assert "absence of error handling is never itself" in lower


def test_rubric_exempts_security():
    assert 'does NOT apply to "security"' in REVIEW_FILE_SYSTEM


def test_critic_prompts_demote_unnamed_robustness_findings():
    for prompt in (CRITIC_BATCHED_SYSTEM, CRITIC_FINDING_SYSTEM):
        assert "defensive-robustness" in prompt
        assert '"speculative"' in prompt
