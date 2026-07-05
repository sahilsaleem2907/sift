"""Tests for src.intelligence.passes.critic."""
from unittest.mock import AsyncMock, patch

import pytest

from src.intelligence.capability import ModelCapability
from src.intelligence.passes.critic import (
    _verification_context,
    critique,
    critique_batched,
    critique_per_finding,
    rule_dedupe,
)
from src.intelligence.effort import EffortLevel, plan_for
from src.intelligence.schema import Certainty, Finding, Impact


def _finding(line: int = 10, impact: Impact = Impact.HIGH, category: str = "correctness") -> Finding:
    return Finding(
        path="app/x.py",
        line=line,
        title="Test issue",
        body="![BUG](https://img.shields.io/badge/BUG-AA0000) issue",
        impact=impact,
        certainty=Certainty.LIKELY,
        category=category,
        origin="llm",
    )


@pytest.mark.asyncio
async def test_batched_keeps_real_bug():
    findings = [_finding()]
    with patch(
        "src.intelligence.passes.critic._call_llm",
        new=AsyncMock(
            return_value='[{"index": 0, "verdict": "keep", "impact": "high", '
            '"certainty": "confirmed", "reason": "valid"}]'
        ),
    ):
        cap = ModelCapability(8192, 2048, False, False)
        kept = await critique_batched(findings, "diff", "title", cap)
    assert len(kept) == 1
    assert kept[0].line == 10


@pytest.mark.asyncio
async def test_batched_drops_false_positive():
    findings = [_finding(impact=Impact.LOW)]
    with patch(
        "src.intelligence.passes.critic._call_llm",
        new=AsyncMock(
            return_value='[{"index": 0, "verdict": "drop", "reason": "style nit"}]'
        ),
    ):
        cap = ModelCapability(8192, 2048, False, False)
        kept = await critique_batched(findings, "diff", "title", cap)
    assert len(kept) == 0


@pytest.mark.asyncio
async def test_batched_missing_verdict_keeps():
    findings = [_finding(), _finding(line=20)]
    with patch(
        "src.intelligence.passes.critic._call_llm",
        new=AsyncMock(return_value="[]"),
    ):
        cap = ModelCapability(8192, 2048, False, False)
        kept = await critique_batched(findings, "diff", "title", cap)
    assert len(kept) == 2


@pytest.mark.asyncio
async def test_critique_empty_input_returns_empty():
    cap = ModelCapability(8192, 2048, False, False)
    plan = plan_for(EffortLevel.BALANCED)
    kept = await critique([], "diff", "title", plan, cap)
    assert kept == []


def test_rule_dedupe_keeps_higher_impact():
    low = _finding(line=5, impact=Impact.LOW)
    high = _finding(line=5, impact=Impact.HIGH)
    result = rule_dedupe([low, high])
    assert len(result) == 1
    assert result[0].impact == Impact.HIGH


# --- verifying critic: context threading + version rule ------------------------

def test_verification_context_empty_without_pr_context():
    assert _verification_context(None) == ""
    assert _verification_context({}) == ""


def test_verification_context_includes_blocks():
    ctx = _verification_context({
        "runtime_target": "Python 3.13 (min of requires-python)",
        "callee_signatures": "def get_stats(): return {'total_items': n}",
    })
    assert "Target runtime: Python 3.13" in ctx
    assert "get_stats" in ctx and "total_items" in ctx


@pytest.mark.asyncio
async def test_per_finding_threads_verification_context_into_prompt():
    """The callee code + runtime target must actually reach the critic's user_content."""
    captured = {}

    async def _fake(system, user_content, **kw):
        captured["user"] = user_content
        return '{"verdict": "keep", "impact": "high", "certainty": "likely", "reason": "ok"}'

    pr_context = {
        "runtime_target": "Python 3.13 (min of requires-python)",
        "callee_signatures": "def get_stats(): return {'total_items': sum(depths)}",
    }
    with patch("src.intelligence.passes.critic._call_llm", new=_fake):
        cap = ModelCapability(8192, 2048, False, False)
        await critique_per_finding([_finding()], "diff", "title", cap, pr_context)
    assert "Verification context" in captured["user"]
    assert "total_items" in captured["user"]
    assert "Target runtime: Python 3.13" in captured["user"]


@pytest.mark.asyncio
async def test_per_finding_drop_honored_with_context():
    with patch(
        "src.intelligence.passes.critic._call_llm",
        new=AsyncMock(return_value='{"verdict": "drop", "reason": "get_stats returns total_items"}'),
    ):
        cap = ModelCapability(8192, 2048, False, False)
        kept = await critique_per_finding(
            [_finding(impact=Impact.LOW)], "diff", "title", cap,
            {"callee_signatures": "def get_stats(): return {'total_items': 0}"},
        )
    assert kept == []


@pytest.mark.asyncio
async def test_per_finding_security_never_dropped():
    """Security findings survive a drop verdict (guard intact after context change)."""
    with patch(
        "src.intelligence.passes.critic._call_llm",
        new=AsyncMock(return_value='{"verdict": "drop", "reason": "x"}'),
    ):
        cap = ModelCapability(8192, 2048, False, False)
        kept = await critique_per_finding(
            [_finding(category="security")], "diff", "title", cap, {"runtime_target": "Python 3.13"},
        )
    assert len(kept) == 1
