"""Tests for src.intelligence.passes.critic."""
from unittest.mock import AsyncMock, patch

import pytest

from sift.intelligence.capability import ModelCapability
from sift.intelligence.passes.critic import (
    critique,
    critique_batched,
    rule_dedupe,
)
from sift.intelligence.effort import EffortLevel, plan_for
from sift.intelligence.schema import Certainty, Finding, Impact


def _finding(line: int = 10, impact: Impact = Impact.HIGH) -> Finding:
    return Finding(
        path="app/x.py",
        line=line,
        title="Test issue",
        body="![BUG](https://img.shields.io/badge/BUG-AA0000) issue",
        impact=impact,
        certainty=Certainty.LIKELY,
        category="correctness",
        origin="llm",
    )


@pytest.mark.asyncio
async def test_batched_keeps_real_bug():
    findings = [_finding()]
    with patch(
        "sift.intelligence.passes.critic._call_llm",
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
        "sift.intelligence.passes.critic._call_llm",
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
        "sift.intelligence.passes.critic._call_llm",
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
