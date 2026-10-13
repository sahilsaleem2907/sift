"""Tests for sift.intelligence.passes.critic."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sift.intelligence.capability import ModelCapability
from sift.intelligence.passes.critic import (
    _CRITIC_TOOL_MAX_STEPS,
    _clamp_certainty_for_critic,
    _critique_with_tools,
    _verification_context,
    critique,
    critique_batched,
    critique_per_finding,
    rule_dedupe,
)
from sift.intelligence.effort import EffortLevel, plan_for
from sift.intelligence.schema import Certainty, Finding, Impact


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
        # The shared PR/diff/verification prefix now arrives via cache_user_prefix (cached for
        # direct Anthropic; concatenated ahead of user_content otherwise). Capture the effective
        # text the model receives — the prefix plus the finding-specific suffix.
        captured["user"] = f"{kw.get('cache_user_prefix') or ''}{user_content}"
        return '{"verdict": "keep", "impact": "high", "certainty": "likely", "reason": "ok"}'

    pr_context = {
        "runtime_target": "Python 3.13 (min of requires-python)",
        "callee_signatures": "def get_stats(): return {'total_items': sum(depths)}",
    }
    with patch("sift.intelligence.passes.critic._call_llm", new=_fake):
        cap = ModelCapability(8192, 2048, False, False)
        await critique_per_finding([_finding()], "diff", "title", cap, pr_context)
    assert "Verification context" in captured["user"]
    assert "total_items" in captured["user"]
    assert "Target runtime: Python 3.13" in captured["user"]


@pytest.mark.asyncio
async def test_per_finding_drop_honored_with_context():
    with patch(
        "sift.intelligence.passes.critic._call_llm",
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
        "sift.intelligence.passes.critic._call_llm",
        new=AsyncMock(return_value='{"verdict": "drop", "reason": "x"}'),
    ):
        cap = ModelCapability(8192, 2048, False, False)
        kept = await critique_per_finding(
            [_finding(category="security")], "diff", "title", cap, {"runtime_target": "Python 3.13"},
        )
    assert len(kept) == 1
@pytest.mark.asyncio
async def test_critic_final_call_keeps_tools_and_disables_tool_choice(monkeypatch):
    """When the critic's tool loop exhausts its step budget, the post-limit final verdict call
    must still declare tools= (the history carries tool_calls, rejected by direct Anthropic
    without it) and set tool_choice='none'. Regression for the martian-benchmark
    UnsupportedParamsError that forced the plain-critic fallback. The critic imports acompletion
    locally from litellm, so we patch it at the source."""
    import litellm

    def _tool_msg():
        tc = MagicMock()
        tc.id = "tc1"
        tc.function = MagicMock()
        tc.function.name = "get_file"
        tc.function.arguments = '{"path": "app/x.py"}'
        m = MagicMock()
        m.content = None
        m.tool_calls = [tc]
        m.role = "assistant"
        return m

    final_msg = MagicMock()
    final_msg.content = '{"verdict": "keep", "reason": "ok"}'
    final_msg.tool_calls = None
    final_msg.role = "assistant"

    calls: list[dict] = []

    async def _record(**kwargs):
        calls.append(kwargs)
        resp = MagicMock()
        # Keep requesting tools until the budget is spent, then answer with a verdict.
        msg = _tool_msg() if len(calls) <= _CRITIC_TOOL_MAX_STEPS else final_msg
        resp.choices = [MagicMock(message=msg)]
        return resp

    monkeypatch.setattr(litellm, "acompletion", _record)

    out = await _critique_with_tools(
        system="sys",
        user_content="u",
        llm_kw={"model": "anthropic/claude-sonnet-4-6", "api_base": None, "api_key": None},
        repo_root="/tmp",
        path_to_content={"app/x.py": "x = 1\n"},
        mod_funcs={},
    )

    assert out == '{"verdict": "keep", "reason": "ok"}'   # final call succeeded, no fallback
    final_kwargs = calls[-1]
    assert final_kwargs.get("tools"), "final call must declare tools= for Anthropic tool history"
    assert final_kwargs.get("tool_choice") == "none"


def _speculative(category: str, impact: Impact = Impact.HIGH) -> Finding:
    f = _finding(impact=impact)
    return Finding(
        path=f.path, line=f.line, title=f.title, body=f.body,
        impact=impact, certainty=Certainty.SPECULATIVE,
        category=category, origin=f.origin,
    )


def test_clamp_leaves_high_correctness_speculative():
    """HIGH-impact correctness findings keep their speculative certainty (PR #24)."""
    out = _clamp_certainty_for_critic([_speculative("correctness")])
    assert out[0].certainty == Certainty.SPECULATIVE


def test_clamp_still_raises_security_speculative():
    out = _clamp_certainty_for_critic([_speculative("security")])
    assert out[0].certainty == Certainty.LIKELY


def test_clamp_raises_concurrency_speculative():
    """A self-downgraded race/TOCTOU finding gets the same clamp as security (no
    single deterministic input, so the generator readily marks it SPECULATIVE)."""
    f = _finding(category="correctness")
    f = Finding(
        path=f.path, line=f.line, title="Race condition on shared cache",
        body="A race condition allows concurrent goroutines to build the index "
             "without synchronization, causing a concurrent map write panic.",
        impact=Impact.HIGH, certainty=Certainty.SPECULATIVE,
        category="correctness", origin=f.origin,
    )
    out = _clamp_certainty_for_critic([f])
    assert out[0].certainty == Certainty.LIKELY


def test_clamp_leaves_unrelated_correctness_speculative():
    """A plain correctness finding with no concurrency language is unaffected."""
    out = _clamp_certainty_for_critic([_speculative("correctness")])
    assert out[0].certainty == Certainty.SPECULATIVE
