"""Tests for src.intelligence.passes.severity."""
from sift.intelligence.effort import EffortLevel, plan_for
from sift.intelligence.passes.severity import (
    _UNVERIFIED_PREFIX,
    apply_final_severity_labels,
    apply_severity_gate,
)
from sift.intelligence.schema import Certainty, Finding, Impact


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
# ---- apply_final_severity_labels ----

_BUG_BODY = (
    "![BUG](https://img.shields.io/badge/BUG-AA0000?style=for-the-badge) "
    "Missing try/catch\n\nAn IO error here may abort shutdown."
)


def _labeled(impact, certainty, category="correctness", body=_BUG_BODY, **kw):
    return Finding(
        path="a.py", line=1, title="Missing try/catch", body=body,
        impact=impact, certainty=certainty, category=category, origin="llm", **kw,
    )


def test_final_labels_swap_bug_to_warning_for_high_speculative():
    out = apply_final_severity_labels([_labeled(Impact.HIGH, Certainty.SPECULATIVE)])
    assert "![WARNING]" in out[0].body
    assert "![BUG]" not in out[0].body


def test_final_labels_keep_bug_for_high_confirmed():
    out = apply_final_severity_labels([_labeled(Impact.HIGH, Certainty.CONFIRMED)])
    assert "![BUG]" in out[0].body


def test_final_labels_preserve_security_badge_when_speculative():
    out = apply_final_severity_labels(
        [_labeled(Impact.HIGH, Certainty.SPECULATIVE, category="security")]
    )
    assert "![SECURITY]" in out[0].body


def test_final_labels_leave_critic_exempt_untouched():
    f = _labeled(Impact.HIGH, Certainty.SPECULATIVE, critic_exempt=True)
    out = apply_final_severity_labels([f])
    assert out[0].body == _BUG_BODY


def test_final_labels_prepend_badge_on_badgeless_body():
    f = _labeled(Impact.MEDIUM, Certainty.LIKELY, body="freeform body text")
    out = apply_final_severity_labels([f])
    assert out[0].body.startswith("![WARNING]")
    assert "freeform body text" in out[0].body


def test_final_labels_idempotent():
    once = apply_final_severity_labels([_labeled(Impact.HIGH, Certainty.SPECULATIVE)])
    twice = apply_final_severity_labels(once)
    assert twice[0].body == once[0].body


def test_final_labels_swap_after_unverified_prefix():
    f = _labeled(Impact.CRITICAL, Certainty.SPECULATIVE, body=_UNVERIFIED_PREFIX + _BUG_BODY)
    out = apply_final_severity_labels([f])
    assert out[0].body.startswith(_UNVERIFIED_PREFIX)
    assert "![BUG]" in out[0].body  # CRITICAL always projects to bug


# --- hedged / self-doubting demoter (impact-aware) ---
_HEDGE = "This may be exploitable if the value is user-controlled."


def test_hedged_low_dropped():
    plan = plan_for(EffortLevel.BALANCED)
    assert apply_severity_gate([_finding(Impact.LOW, Certainty.LIKELY, body=_HEDGE)], plan) == []


def test_hedged_medium_dropped():
    plan = plan_for(EffortLevel.BALANCED)
    f = _finding(Impact.MEDIUM, Certainty.LIKELY, body="Cannot be confirmed from the diff.")
    assert apply_severity_gate([f], plan) == []


def test_hedged_high_kept_and_demoted():
    plan = plan_for(EffortLevel.BALANCED)
    out = apply_severity_gate([_finding(Impact.HIGH, Certainty.LIKELY, body=_HEDGE)], plan)
    assert len(out) == 1 and out[0].certainty == Certainty.SPECULATIVE


def test_hedged_critical_kept_as_unverified():
    plan = plan_for(EffortLevel.BALANCED)
    f = _finding(Impact.CRITICAL, Certainty.LIKELY, body="RCE if the input is attacker-controlled.")
    out = apply_severity_gate([f], plan)
    assert len(out) == 1 and out[0].body.startswith(_UNVERIFIED_PREFIX)


def test_self_negation_dropped():
    plan = plan_for(EffortLevel.BALANCED)
    f = _finding(Impact.MEDIUM, Certainty.LIKELY, body="On closer look this is not a bug.")
    assert apply_severity_gate([f], plan) == []


def test_logical_conditional_not_hedged():
    # "if x is null -> crash" is a real bug trigger, not an epistemic hedge: must be kept.
    plan = plan_for(EffortLevel.BALANCED)
    f = _finding(Impact.LOW, Certainty.LIKELY, body="If x is null this dereferences and crashes.")
    assert len(apply_severity_gate([f], plan)) == 1
