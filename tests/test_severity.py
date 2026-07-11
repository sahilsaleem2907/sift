"""Tests for src.intelligence.passes.severity."""
from sift.intelligence.effort import EffortLevel, plan_for
from sift.intelligence.passes.severity import apply_severity_gate, _UNVERIFIED_PREFIX
from sift.intelligence.schema import Certainty, Finding, Impact


def _finding(impact: Impact, certainty: Certainty, body: str = "body") -> Finding:
    return Finding(
        path="a.py",
        line=1,
        title="t",
        body=body,
        impact=impact,
        certainty=certainty,
        category="correctness",
        origin="llm",
    )


def test_trivial_dropped():
    plan = plan_for(EffortLevel.BALANCED)
    out = apply_severity_gate([_finding(Impact.TRIVIAL, Certainty.CONFIRMED)], plan)
    assert out == []


def test_speculative_low_dropped():
    plan = plan_for(EffortLevel.BALANCED)
    out = apply_severity_gate([_finding(Impact.LOW, Certainty.SPECULATIVE)], plan)
    assert out == []


def test_speculative_medium_kept():
    plan = plan_for(EffortLevel.BALANCED)
    out = apply_severity_gate([_finding(Impact.MEDIUM, Certainty.SPECULATIVE)], plan)
    assert len(out) == 1


def test_speculative_critical_kept_as_unverified():
    plan = plan_for(EffortLevel.BALANCED)
    out = apply_severity_gate([_finding(Impact.CRITICAL, Certainty.SPECULATIVE)], plan)
    assert len(out) == 1
    assert out[0].body.startswith(_UNVERIFIED_PREFIX)


def test_confirmed_high_kept():
    plan = plan_for(EffortLevel.BALANCED)
    f = _finding(Impact.HIGH, Certainty.CONFIRMED, body="plain body")
    out = apply_severity_gate([f], plan)
    assert len(out) == 1
    assert out[0].body == "plain body"
