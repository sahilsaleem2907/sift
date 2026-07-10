"""Tests for candidate category resolution (explicit field vs badge inference)."""
from src.intelligence.passes.candidates import resolve_category


def test_explicit_category_used():
    # explicit design category wins even though the badge would infer correctness
    c = {"category": "design", "body": "![WARNING](x) reaches into internals"}
    assert resolve_category(c) == "design"


def test_invalid_explicit_falls_back_to_badge():
    # WARNING badge → correctness via _BADGE_TO_CATEGORY
    c = {"category": "not-a-category", "body": "![WARNING](x) something"}
    assert resolve_category(c) == "correctness"


def test_missing_explicit_uses_badge_suggestion_maintainability():
    c = {"body": "![SUGGESTION](x) rename this"}
    assert resolve_category(c) == "maintainability"


def test_explicit_maintainability_over_correctness_badge():
    # the exact bug the field fixes: a WARNING-badged coupling note self-labels maintainability
    c = {"category": "maintainability", "body": "![WARNING](x) duplicated logic"}
    assert resolve_category(c) == "maintainability"
