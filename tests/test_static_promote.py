"""Tests for static_promote._build_finding title enrichment."""
from unittest import mock

import pytest

from src.intelligence.passes.static_promote import (
    _build_finding,
    promote_static_findings,
    should_auto_promote,
)
from src.intelligence.schema import Impact, Certainty


_PYRIGHT_FINDING = {
    "check_id": "pyright/reportAttributeAccessIssue",
    "line": 5,
    "severity": "ERROR",
    "message": 'Cannot access attribute "shutdown" for class "Queue"',
}


def test_pyright_error_is_auto_promoted():
    assert should_auto_promote(_PYRIGHT_FINDING) is True


@pytest.mark.asyncio
async def test_pyright_finding_promoted_critic_exempt():
    async def _fake_enrich(raw, path, origin, diff):
        return [{"body": f.get("message", ""), "fix": "", "title": ""} for f in raw]

    with mock.patch("src.intelligence.passes.static_promote._enrich_batch", _fake_enrich):
        out = await promote_static_findings(
            "pkg/mod.py", "@@ +5 @@\n+ q.shutdown()",
            semgrep_findings=[], codeql_findings=[], pyright_findings=[_PYRIGHT_FINDING],
        )
    assert len(out) == 1
    assert out[0].origin == "pyright"
    assert out[0].critic_exempt is True
    assert out[0].line == 5


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
