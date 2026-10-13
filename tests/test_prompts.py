"""Prompt invariants: shared critic rules, confidence bands that match the code, no contradictions."""
from sift.intelligence import prompts
from sift.intelligence.prompts import (
    CRITIC_BATCHED_SYSTEM,
    CRITIC_FINDING_SYSTEM,
    REVIEW_FILE_SYSTEM,
    REVIEW_FILE_SYSTEM_EXTENDED_THINKING,
    TEST_FILE_APPENDIX,
)
from sift.intelligence.schema import Certainty, confidence_to_certainty

_SHARED_CRITIC_FRAGMENTS = (
    "_DROP_CONTRADICTED_BY_DIFF",
    "_DROP_UNCHANGED_CODE",
    "_DROP_DISPROVED_BY_CONTEXT",
    "_DROP_TARGET_RUNTIME",
    "_DROP_DESIGN_NOISE",
    "_DROP_SELF_CONTRADICTION",
    "_EVIDENCE_RULES",
    "_KEEP_RULES",
    "_CERTAINTY_RULE",
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_both_critic_prompts_carry_every_shared_rule():
    for name in _SHARED_CRITIC_FRAGMENTS:
        fragment = _flat(getattr(prompts, name))
        assert fragment in _flat(CRITIC_BATCHED_SYSTEM), f"{name} missing from batched critic"
        assert fragment in _flat(CRITIC_FINDING_SYSTEM), f"{name} missing from per-finding critic"


def test_critic_downgrades_unverified_instead_of_dropping():
    for prompt in (CRITIC_BATCHED_SYSTEM, CRITIC_FINDING_SYSTEM):
        flat = _flat(prompt)
        assert "still could not CONFIRM it, KEEP with certainty=\"speculative\"" in flat


def test_critic_drops_hedged_self_contradiction():
    for prompt in (CRITIC_BATCHED_SYSTEM, CRITIC_FINDING_SYSTEM):
        assert "works, but is fragile" in prompt
        assert "correct only if" in prompt


def test_critic_rule_numbering_is_sequential():
    import re

    for prompt in (CRITIC_BATCHED_SYSTEM, CRITIC_FINDING_SYSTEM):
        numbers = [int(n) for n in re.findall(r"^(\d+)\. ", prompt, flags=re.M)]
        assert numbers == list(range(1, len(numbers) + 1))


def test_generator_confidence_bands_match_code():
    flat = _flat(REVIEW_FILE_SYSTEM)
    assert "8-10 = confirmed; 7 = likely; 5-6 = speculative (still reported); 1-4 = omit" in flat
    assert confidence_to_certainty(8) == Certainty.CONFIRMED
    assert confidence_to_certainty(7) == Certainty.LIKELY
    assert confidence_to_certainty(6) == Certainty.SPECULATIVE
    assert confidence_to_certainty(5) == Certainty.SPECULATIVE
    # the old contradictory band ("6-7 = likely") is gone everywhere
    assert "6-7" not in REVIEW_FILE_SYSTEM


def test_generator_has_no_contradictory_coupling_bullet():
    # A dangling "Coupling or abstraction violations" bullet used to invite exactly the
    # layering findings the checklist above it says are NOT findings.
    assert "Coupling or abstraction violations" not in REVIEW_FILE_SYSTEM


def test_generator_security_certainty_requires_established_premise():
    assert "the input really is attacker-controlled AND really reaches the sink" in _flat(REVIEW_FILE_SYSTEM)


def test_test_hygiene_is_maintainability():
    assert "Test hygiene is ALWAYS \"maintainability\"" in REVIEW_FILE_SYSTEM
    assert "is \"maintainability\"" in TEST_FILE_APPENDIX


def test_extended_thinking_variant_still_strips_visible_reasoning():
    assert "<reasoning>" in REVIEW_FILE_SYSTEM
    assert "<reasoning>" not in REVIEW_FILE_SYSTEM_EXTENDED_THINKING
