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
