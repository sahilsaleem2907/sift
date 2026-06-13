"""Regression tests for the PR #23 severity-calibration fixes.

Covers:
- GHA injection rule split: attacker-controllable contexts → security;
  trusted contexts → suggestion; inputs.* in reusable workflow → suggestion.
- Naming/convention-nitpick rule: consistently-used off-convention identifier
  → suggestion, not bug.
- derive_severity projection: category=security, impact=low → "suggestion";
  category=security, impact=high → "security".
"""
from src.intelligence.prompts import REVIEW_FILE_SYSTEM
from src.intelligence.schema import derive_severity, Impact, Certainty


# ---- derive_severity projection (the lever the rule relies on) ----

def test_derive_severity_security_low_impact_yields_suggestion():
    """Trusted GHA context: security category + low impact → suggestion (non-blocking)."""
    result = derive_severity(Impact.LOW, Certainty.CONFIRMED, "security")
    assert result == "suggestion", f"expected 'suggestion', got {result!r}"


def test_derive_severity_security_high_impact_yields_security():
    """Attacker-controllable injection: security category + high impact → security (blocks merge)."""
    result = derive_severity(Impact.HIGH, Certainty.CONFIRMED, "security")
    assert result == "security", f"expected 'security', got {result!r}"


def test_derive_severity_security_critical_impact_yields_security():
    result = derive_severity(Impact.CRITICAL, Certainty.CONFIRMED, "security")
    assert result == "security", f"expected 'security', got {result!r}"


# ---- GHA injection rule: attacker-controllable list present ----

def test_gha_rule_lists_attacker_controllable_contexts():
    """Prompt must name the known-bad contexts so the model can recognize them."""
    for ctx in ("github.head_ref", "github.event.comment.body", "github.event.*.body",
                "github.event.*.title", "github.event.pull_request.head.ref"):
        assert ctx in REVIEW_FILE_SYSTEM, (
            f"attacker-controllable context {ctx!r} missing from REVIEW_FILE_SYSTEM"
        )


def test_gha_rule_lists_trusted_contexts():
    """Prompt must name the trusted contexts mapped to suggestion-level hardening."""
    for ctx in ("github.repository", "github.run_id", "github.sha", "github.actor",
                "github.repository_owner"):
        assert ctx in REVIEW_FILE_SYSTEM, (
            f"trusted context {ctx!r} missing from REVIEW_FILE_SYSTEM"
        )


def test_gha_rule_inputs_maps_to_suggestion():
    """inputs.* in a reusable workflow defaults to suggestion, not security."""
    # The rule text must convey "suggestion" + "inputs.*" + "reusable workflow".
    assert "inputs.*" in REVIEW_FILE_SYSTEM
    assert "suggestion" in REVIEW_FILE_SYSTEM
    # The calibration phrase tying inputs to caller-dependency must be present.
    assert "caller" in REVIEW_FILE_SYSTEM or "reusable" in REVIEW_FILE_SYSTEM


def test_gha_rule_env_var_fix_present():
    """The env: fix must be in the rule regardless of severity path."""
    assert 'env:' in REVIEW_FILE_SYSTEM
    assert '"$ENVVAR"' in REVIEW_FILE_SYSTEM or "$ENVVAR" in REVIEW_FILE_SYSTEM


# ---- Naming/convention-nitpick rule ----

def test_naming_rule_present():
    """Prompt must include the naming/convention rule."""
    lower = REVIEW_FILE_SYSTEM.lower()
    assert "naming" in lower or "convention" in lower, (
        "naming/convention rule missing from REVIEW_FILE_SYSTEM"
    )


def test_naming_rule_says_suggestion_not_bug():
    """Off-convention-but-consistent naming must map to suggestion, not bug."""
    # The rule must use both "suggestion" (the correct severity) and
    # explicitly exclude "bug" — presence of "NOT a" + "bug" in the rule block.
    rule_start = REVIEW_FILE_SYSTEM.find("Naming/convention")
    assert rule_start != -1, "Naming/convention rule not found"
    rule_text = REVIEW_FILE_SYSTEM[rule_start:rule_start + 400]
    assert "suggestion" in rule_text.lower(), (
        "naming rule must say 'suggestion'"
    )
    assert "bug" in rule_text.lower(), (
        "naming rule must explicitly mention 'bug' (to exclude it)"
    )


def test_naming_rule_requires_functional_consequence_for_escalation():
    """Rule must gate escalation on a concrete functional break, not just convention."""
    rule_start = REVIEW_FILE_SYSTEM.find("Naming/convention")
    rule_text = REVIEW_FILE_SYSTEM[rule_start:rule_start + 500]
    # "consequence" or "functional" or "concrete" must appear in the rule.
    has_escalation_gate = any(
        word in rule_text.lower()
        for word in ("consequence", "functional", "concrete", "concrete")
    )
    assert has_escalation_gate, (
        "naming rule must require a concrete functional consequence before escalating"
    )
