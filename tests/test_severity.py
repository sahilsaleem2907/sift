"""Tests for src.intelligence.passes.severity."""
from src.intelligence.effort import EffortLevel, plan_for
from src.intelligence.passes.severity import apply_severity_gate, _UNVERIFIED_PREFIX
from src.intelligence.schema import Certainty, Finding, Impact


def _finding(
    impact: Impact,
    certainty: Certainty,
    body: str = "body",
    category: str = "correctness",
    critic_exempt: bool = False,
) -> Finding:
    return Finding(
        path="a.py",
        line=1,
        title="t",
        body=body,
        impact=impact,
        certainty=certainty,
        category=category,
        origin="llm",
        critic_exempt=critic_exempt,
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


# --- confirmed-only gate for design/maintainability/style (Increment 2 / G) ---

def test_design_likely_dropped():
    plan = plan_for(EffortLevel.BALANCED)
    # a high-impact but non-confirmed design finding is noise → dropped
    out = apply_severity_gate(
        [_finding(Impact.HIGH, Certainty.LIKELY, category="design")], plan
    )
    assert out == []


def test_maintainability_likely_dropped():
    plan = plan_for(EffortLevel.BALANCED)
    out = apply_severity_gate(
        [_finding(Impact.MEDIUM, Certainty.LIKELY, category="maintainability")], plan
    )
    assert out == []


def test_design_confirmed_kept():
    plan = plan_for(EffortLevel.BALANCED)
    out = apply_severity_gate(
        [_finding(Impact.MEDIUM, Certainty.CONFIRMED, category="design")], plan
    )
    assert len(out) == 1


def test_correctness_likely_not_gated_by_category():
    plan = plan_for(EffortLevel.BALANCED)
    # correctness is unaffected by the category gate — survives at LIKELY
    out = apply_severity_gate(
        [_finding(Impact.HIGH, Certainty.LIKELY, category="correctness")], plan
    )
    assert len(out) == 1


def test_design_critic_exempt_bypasses_category_gate():
    plan = plan_for(EffortLevel.BALANCED)
    # a static-tool (critic_exempt) finding is pre-confirmed and bypasses all filtering
    out = apply_severity_gate(
        [_finding(Impact.LOW, Certainty.LIKELY, category="design", critic_exempt=True)],
        plan,
    )
    assert len(out) == 1
