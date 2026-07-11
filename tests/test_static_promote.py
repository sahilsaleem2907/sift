"""Tests for static_promote._build_finding title enrichment."""
from sift.intelligence.passes.static_promote import _build_finding
from sift.intelligence.schema import Impact, Certainty


_SECRET_FINDING = {
    "check_id": "builtin.hardcoded-secret.github-pat",
    "line": 25,
    "severity": "ERROR",
    "message": "Hardcoded secret detected.",
}

_SEMGREP_ERROR_FINDING = {
    "check_id": "semgrep.parse-error",
    "line": 10,
    "severity": "ERROR",
    "message": "Syntax error.",
}


def test_enriched_title_used_when_provided():
    finding = _build_finding(
        _SECRET_FINDING, "src/extension.ts", "semgrep",
        body="A hardcoded GitHub PAT was found.",
        fix="Remove the token.",
        title="Hardcoded GitHub token",
    )
    assert finding.title == "Hardcoded GitHub token"
    assert "Hardcoded GitHub token" in finding.body


def test_fallback_to_rule_id_last_segment_when_title_empty():
    finding = _build_finding(
        _SECRET_FINDING, "src/extension.ts", "semgrep",
        body="A hardcoded GitHub PAT was found.",
        fix="",
        title="",
    )
    # Falls back to last segment of check_id
    assert finding.title == "github-pat"
    assert "github-pat" in finding.body


def test_fallback_to_rule_id_last_segment_when_title_omitted():
    finding = _build_finding(
        _SEMGREP_ERROR_FINDING, "src/extension.ts", "semgrep",
        body="Syntax error in the file.",
        fix="",
    )
    assert finding.title == "parse-error"


def test_title_capped_at_60_chars():
    long_title = "A" * 80
    finding = _build_finding(
        _SECRET_FINDING, "src/extension.ts", "semgrep",
        body="body", fix="", title=long_title,
    )
    assert len(finding.title) <= 60


def test_critic_exempt_and_impact_unaffected_by_title():
    finding = _build_finding(
        _SECRET_FINDING, "src/extension.ts", "semgrep",
        body="body", fix="", title="Hardcoded GitHub token",
    )
    assert finding.critic_exempt is True
    assert finding.impact == Impact.CRITICAL
    assert finding.certainty == Certainty.CONFIRMED
    assert finding.category == "security"


def test_security_badge_present_in_body():
    finding = _build_finding(
        _SECRET_FINDING, "src/extension.ts", "semgrep",
        body="A secret was found.", fix="", title="Hardcoded GitHub token",
    )
    assert "SECURITY" in finding.body
