"""Tests for src.intelligence.passes.severity."""
from sift.intelligence.effort import EffortLevel, plan_for
from sift.intelligence.passes.severity import (
    _UNVERIFIED_PREFIX,
    apply_final_severity_labels,
    apply_severity_gate,
)
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
