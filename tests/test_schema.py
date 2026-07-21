"""Tests for src.intelligence.schema."""
from sift.intelligence.schema import (
    Certainty,
    Finding,
    Impact,
    confidence_to_certainty,
    derive_severity,
    from_legacy_item,
)


def test_derive_severity_security_high():
    assert derive_severity(Impact.HIGH, Certainty.CONFIRMED, "security") == "security"


def test_derive_severity_critical_speculative():
    assert derive_severity(Impact.CRITICAL, Certainty.SPECULATIVE, "correctness") == "bug"


def test_derive_severity_high_speculative():
    assert derive_severity(Impact.HIGH, Certainty.SPECULATIVE, "correctness") == "warning"


def test_derive_severity_trivial():
    assert derive_severity(Impact.TRIVIAL, Certainty.CONFIRMED, "style") == "informational"


def test_confidence_to_certainty():
    assert confidence_to_certainty(9) == Certainty.CONFIRMED
    assert confidence_to_certainty(7) == Certainty.LIKELY
    assert confidence_to_certainty(5) == Certainty.SPECULATIVE


def test_legacy_mapping_round_trip_severity():
    for old_sev in ("bug", "security", "warning", "suggestion", "informational"):
        item = {
            "line": 10,
            "severity": old_sev,
            "title": "Test",
            "body": "desc",
            "confidence": 9,
        }
        f = from_legacy_item(item, "a.py", "formatted body")
        if old_sev == "security":
            assert f.severity() == "security"
        elif old_sev == "bug":
            assert f.severity() == "bug"
        elif old_sev == "warning":
            assert f.severity() == "warning"
        elif old_sev == "suggestion":
            assert f.severity() == "suggestion"
        else:
            assert f.severity() == "informational"


def test_from_legacy_item_confidence_boundaries():
    """Confidence maps to certainty: 8+ confirmed, 7 likely, <=6 speculative."""
    expected = {
        5: Certainty.SPECULATIVE,
        6: Certainty.SPECULATIVE,
        7: Certainty.LIKELY,
        8: Certainty.CONFIRMED,
    }
    for conf, certainty in expected.items():
        item = {"line": 1, "severity": "bug", "title": "t", "confidence": conf}
        f = from_legacy_item(item, "a.py", "body")
        assert f.certainty == certainty, f"confidence={conf}"
        assert f.impact == Impact.HIGH


def test_from_legacy_item_bug_low_confidence_is_warning():
    """The PR #24 regression lever: a shaky 'bug' projects to warning, not bug."""
    item = {"line": 1, "severity": "bug", "title": "Missing try/catch", "confidence": 6}
    f = from_legacy_item(item, "a.py", "body")
    assert f.severity() == "warning"


def test_from_legacy_item_origin_and_post_inline():
    item = {"line": 1, "severity": "warning", "title": "t", "confidence": 9}
    f = from_legacy_item(item, "a.py", "body", origin="agentic", post_inline=False)
    assert f.origin == "agentic"
    assert f.post_inline is False


def test_to_comment_dict():
    f = Finding(
        path="x.py",
        line=5,
        title="T",
        body="B",
        impact=Impact.LOW,
        certainty=Certainty.LIKELY,
        category="maintainability",
        origin="llm",
    )
    d = f.to_comment_dict()
    assert set(d.keys()) == {"path", "line", "body", "post_inline"}
    assert d["path"] == "x.py"
    assert d["line"] == 5
    assert d["body"] == "B"
