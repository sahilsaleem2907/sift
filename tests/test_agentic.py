"""Tests for Phase 4 agentic review loop."""
import json
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from sift import config
from sift.intelligence.ast.function_extract import FunctionChunk
from sift.intelligence.capability import ModelCapability
from sift.intelligence.effort import EffortLevel, plan_for
from sift.intelligence.passes.agentic import TOOLS, agentic_review
from sift.intelligence.passes.pipeline import FileReviewInput


def _mock_message(content=None, tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"
    return msg


def _mock_tool_call(call_id: str, name: str, arguments: str):
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _mock_emit_message(findings: list):
    """A response where the model submits findings via the forced emit_findings tool."""
    return _mock_message(
        tool_calls=[
            _mock_tool_call("emit1", "emit_findings", json.dumps({"findings": findings}))
        ]
    )


def _resp(message):
    r = MagicMock()
    r.choices = [MagicMock(message=message)]
    return r


@pytest.mark.asyncio
async def test_agentic_tool_call_resolved():
    plan = plan_for(EffortLevel.HIGH)
    cap = ModelCapability(128_000, 4096, True, False)
    file_input = FileReviewInput(
        path="app/api.py",
        file_diff="diff --git a/app/api.py\n+++ b/app/api.py\n@@ -1 +1,2 @@\n+x=1\n",
        pr_context={},
    )
    path_to_content = {"app/util.py": "def helper():\n    return None\n"}
    mod_funcs = {
        "app/util.py": [
            FunctionChunk(
                path="app/util.py",
                name="helper",
                start_line=1,
                end_line=2,
                text="def helper():\n    return None",
                content_hash="h",
            )
        ]
    }

    tool_msg = _mock_message(
        tool_calls=[
            _mock_tool_call(
                "tc1",
                "get_function",
                '{"path": "app/util.py", "name": "helper"}',
            )
        ]
    )
    final_msg = _mock_message(
        content='[{"line": 1, "severity": "bug", "title": "Null return", '
        '"body": "helper returns None", "confidence": 9}]'
    )

    resp_tool = MagicMock()
    resp_tool.choices = [MagicMock(message=tool_msg)]
    resp_final = MagicMock()
    resp_final.choices = [MagicMock(message=final_msg)]

    with mock.patch(
        "sift.intelligence.passes.agentic.acompletion",
        new=AsyncMock(side_effect=[resp_tool, resp_final]),
    ):
        findings = await agentic_review(
            file_input, plan, cap, path_to_content, mod_funcs
        )

    assert len(findings) == 1
    assert findings[0].origin == "agentic"
    assert findings[0].line == 1


@pytest.mark.asyncio
async def test_agentic_respects_step_cap():
    plan = plan_for(EffortLevel.HIGH)
    cap = ModelCapability(128_000, 4096, True, False)
    file_input = FileReviewInput(
        path="app/a.py",
        file_diff="diff --git a/app/a.py\n+++ b/app/a.py\n@@ -1 +1,2 @@\n+x\n",
        pr_context={},
    )

    always_tool = _mock_message(
        tool_calls=[_mock_tool_call("tc1", "get_file", '{"path": "app/a.py"}')]
    )
    # Budget-exhausted final answer now arrives via the forced emit_findings tool.
    resp_tool = _resp(always_tool)
    resp_final = _resp(_mock_emit_message([]))

    old_max = config.SIFT_AGENTIC_MAX_STEPS
    config.SIFT_AGENTIC_MAX_STEPS = 2
    try:
        with mock.patch(
            "sift.intelligence.passes.agentic.acompletion",
            new=AsyncMock(side_effect=[resp_tool, resp_tool, resp_final]),
        ) as mock_llm:
            findings = await agentic_review(file_input, plan, cap, {"app/a.py": "x=1\n"})
            assert mock_llm.call_count == 3
            assert findings == []
    finally:
        config.SIFT_AGENTIC_MAX_STEPS = old_max


@pytest.mark.asyncio
async def test_agentic_final_call_forces_emit_findings():
    """When the step budget is exhausted, the post-limit final call forces the emit_findings
    tool so the answer is schema-constrained JSON (deterministic, always parseable). It must
    still declare the prior read-tools alongside emit_findings (the message history carries
    tool_calls, which direct Anthropic rejects without tools=)."""
    plan = plan_for(EffortLevel.HIGH)
    cap = ModelCapability(128_000, 4096, True, False)
    file_input = FileReviewInput(
        path="app/a.py",
        file_diff="diff --git a/app/a.py\n+++ b/app/a.py\n@@ -1 +1,2 @@\n+x\n",
        pr_context={},
    )
    always_tool = _mock_message(
        tool_calls=[_mock_tool_call("tc1", "get_file", '{"path": "app/a.py"}')]
    )
    emit_msg = _mock_emit_message([])

    calls: list[dict] = []

    async def _record(**kwargs):
        calls.append(kwargs)
        # Keep requesting tools until the budget is spent, then emit findings.
        msg = always_tool if len(calls) <= config.SIFT_AGENTIC_MAX_STEPS else emit_msg
        return _resp(msg)

    old_max = config.SIFT_AGENTIC_MAX_STEPS
    config.SIFT_AGENTIC_MAX_STEPS = 2
    try:
        with mock.patch("sift.intelligence.passes.agentic.acompletion", new=_record):
            findings = await agentic_review(file_input, plan, cap, {"app/a.py": "x=1\n"})
    finally:
        config.SIFT_AGENTIC_MAX_STEPS = old_max

    assert findings == []            # forced emit succeeded (no fallback), returned []
    final_kwargs = calls[-1]
    tool_names = {t["function"]["name"] for t in final_kwargs.get("tools", [])}
    assert "emit_findings" in tool_names
    assert "get_file" in tool_names, "prior read-tools must stay declared for Anthropic tool history"
    assert final_kwargs.get("tool_choice") == {
        "type": "function", "function": {"name": "emit_findings"}
    }


@pytest.mark.asyncio
async def test_agentic_non_tool_model_falls_back_to_generate_candidates():
    plan = plan_for(EffortLevel.HIGH)
    cap = ModelCapability(8192, 2048, False, False)
    file_input = FileReviewInput(path="app/a.py", file_diff="+x\n", pr_context={})

    with mock.patch(
        "sift.intelligence.passes.agentic.agentic_review",
        new=AsyncMock(),
    ) as mock_agentic:
        from sift.intelligence.passes.pipeline import run_pipeline_per_file

        with mock.patch(
            "sift.intelligence.passes.candidates.llm_client.review_file",
            new=AsyncMock(return_value=[]),
        ):
            await run_pipeline_per_file(file_input, "t", plan, cap, None)
        mock_agentic.assert_not_called()


@pytest.mark.asyncio
async def test_agentic_error_falls_back():
    plan = plan_for(EffortLevel.HIGH)
    cap = ModelCapability(128_000, 4096, True, False)
    file_input = FileReviewInput(
        path="app/a.py",
        file_diff="diff --git a/app/a.py\n+++ b/app/a.py\n@@ -1 +1,2 @@\n+x\n",
        pr_context={},
    )

    with mock.patch(
        "sift.intelligence.passes.agentic.acompletion",
        new=AsyncMock(side_effect=RuntimeError("api down")),
    ):
        with mock.patch(
            "sift.intelligence.passes.agentic.generate_candidates",
            new=AsyncMock(return_value=[]),
        ) as mock_gen:
            findings = await agentic_review(file_input, plan, cap, {})
            mock_gen.assert_called_once()
            assert findings == []


@pytest.mark.asyncio
async def test_agentic_error_finish_reason_retries_then_succeeds():
    """finish_reason='error' with empty content must be retried, not treated as
    'no findings' (the flusher.py failure in the PR #93824 benchmark run)."""
    plan = plan_for(EffortLevel.HIGH)
    cap = ModelCapability(128_000, 4096, True, False)
    file_input = FileReviewInput(
        path="app/a.py",
        file_diff="diff --git a/app/a.py\n+++ b/app/a.py\n@@ -1 +1,2 @@\n+x\n",
        pr_context={},
    )

    err_msg = _mock_message(content="")
    resp_err = MagicMock()
    resp_err.choices = [MagicMock(message=err_msg, finish_reason="error")]
    final_msg = _mock_message(
        content='[{"line": 1, "severity": "bug", "title": "x", "body": "y", "confidence": 9}]'
    )
    resp_final = MagicMock()
    resp_final.choices = [MagicMock(message=final_msg, finish_reason="stop")]

    with mock.patch("sift.intelligence.passes.agentic.asyncio.sleep", new=AsyncMock()):
        with mock.patch(
            "sift.intelligence.passes.agentic.acompletion",
            new=AsyncMock(side_effect=[resp_err, resp_final]),
        ) as mock_llm:
            findings = await agentic_review(file_input, plan, cap, {})

    assert mock_llm.call_count == 2          # retried past the error
    assert len(findings) == 1
    assert findings[0].line == 1


@pytest.mark.asyncio
async def test_agentic_forced_emit_returns_findings():
    """Tier 1: the forced emit_findings tool call is parsed deterministically into findings."""
    plan = plan_for(EffortLevel.HIGH)
    cap = ModelCapability(128_000, 4096, True, False)
    file_input = FileReviewInput(
        path="app/a.py",
        file_diff="diff --git a/app/a.py\n+++ b/app/a.py\n@@ -1 +1,2 @@\n+x\n",
        pr_context={},
    )
    always_tool = _mock_message(
        tool_calls=[_mock_tool_call("tc1", "get_file", '{"path": "app/a.py"}')]
    )
    emit = _mock_emit_message([
        {"line": 1, "severity": "bug", "title": "Bad", "body": "boom", "confidence": 9},
        {"line": 2, "severity": "nit", "title": "Weak", "body": "meh", "confidence": 3},
    ])

    old_max = config.SIFT_AGENTIC_MAX_STEPS
    config.SIFT_AGENTIC_MAX_STEPS = 1
    try:
        with mock.patch(
            "sift.intelligence.passes.agentic.acompletion",
            new=AsyncMock(side_effect=[_resp(always_tool), _resp(emit)]),
        ):
            findings = await agentic_review(file_input, plan, cap, {"app/a.py": "x=1\n"})
    finally:
        config.SIFT_AGENTIC_MAX_STEPS = old_max

    # confidence<5 item is filtered by the shared _items_to_comments path
    assert len(findings) == 1
    assert findings[0].line == 1
    assert findings[0].origin == "agentic"


@pytest.mark.asyncio
async def test_agentic_tier2_fallback_when_forced_emit_errors():
    """Tier 2: if the forced emit call errors (e.g. endpoint rejects forced tool_choice),
    fall back to the free-text final call routed through parse_with_repair."""
    plan = plan_for(EffortLevel.HIGH)
    cap = ModelCapability(128_000, 4096, True, False)
    file_input = FileReviewInput(
        path="app/a.py",
        file_diff="diff --git a/app/a.py\n+++ b/app/a.py\n@@ -1 +1,2 @@\n+x\n",
        pr_context={},
    )
    always_tool = _mock_message(
        tool_calls=[_mock_tool_call("tc1", "get_file", '{"path": "app/a.py"}')]
    )
    free_text = _mock_message(
        content='[{"line": 1, "severity": "bug", "title": "x", "body": "y", "confidence": 9}]'
    )

    calls = {"n": 0}

    async def _side(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp(always_tool)
        if calls["n"] == 2:  # the forced emit_findings call
            raise RuntimeError("UnsupportedParamsError: forced tool_choice not supported")
        return _resp(free_text)  # tier-2 free-text final call

    old_max = config.SIFT_AGENTIC_MAX_STEPS
    config.SIFT_AGENTIC_MAX_STEPS = 1
    try:
        with mock.patch("sift.intelligence.passes.agentic.acompletion", new=_side):
            findings = await agentic_review(file_input, plan, cap, {"app/a.py": "x=1\n"})
    finally:
        config.SIFT_AGENTIC_MAX_STEPS = old_max

    assert len(findings) == 1
    assert findings[0].line == 1


@pytest.mark.asyncio
async def test_agentic_mid_loop_empty_makes_no_extra_call():
    """A legit empty [] emitted mid-loop is authoritative — no forced-emit repair call."""
    plan = plan_for(EffortLevel.HIGH)
    cap = ModelCapability(128_000, 4096, True, False)
    file_input = FileReviewInput(
        path="app/a.py",
        file_diff="diff --git a/app/a.py\n+++ b/app/a.py\n@@ -1 +1,2 @@\n+x\n",
        pr_context={},
    )
    empty = _mock_message(content="[]")

    with mock.patch(
        "sift.intelligence.passes.agentic.acompletion",
        new=AsyncMock(side_effect=[_resp(empty)]),
    ) as mock_llm:
        findings = await agentic_review(file_input, plan, cap, {"app/a.py": "x=1\n"})

    assert findings == []
    assert mock_llm.call_count == 1  # no extra forced-emit call


@pytest.mark.asyncio
async def test_agentic_mid_loop_parse_failure_triggers_emit():
    """Mid-loop prose with no extractable JSON array triggers a forced-emit repair call."""
    plan = plan_for(EffortLevel.HIGH)
    cap = ModelCapability(128_000, 4096, True, False)
    file_input = FileReviewInput(
        path="app/a.py",
        file_diff="diff --git a/app/a.py\n+++ b/app/a.py\n@@ -1 +1,2 @@\n+x\n",
        pr_context={},
    )
    prose = _mock_message(content="After reviewing, I could not produce structured output.")
    emit = _mock_emit_message([
        {"line": 1, "severity": "bug", "title": "x", "body": "y", "confidence": 8},
    ])

    with mock.patch(
        "sift.intelligence.passes.agentic.acompletion",
        new=AsyncMock(side_effect=[_resp(prose), _resp(emit)]),
    ) as mock_llm:
        findings = await agentic_review(file_input, plan, cap, {"app/a.py": "x=1\n"})

    assert mock_llm.call_count == 2  # prose step + forced-emit repair
    assert len(findings) == 1
    assert findings[0].line == 1


@pytest.mark.asyncio
async def test_agentic_error_finish_reason_exhausts_to_fallback():
    """Persistent finish_reason='error' falls back to generate_candidates."""
    plan = plan_for(EffortLevel.HIGH)
    cap = ModelCapability(128_000, 4096, True, False)
    file_input = FileReviewInput(
        path="app/a.py",
        file_diff="diff --git a/app/a.py\n+++ b/app/a.py\n@@ -1 +1,2 @@\n+x\n",
        pr_context={},
    )

    err_msg = _mock_message(content="")
    resp_err = MagicMock()
    resp_err.choices = [MagicMock(message=err_msg, finish_reason="error")]

    with mock.patch("sift.intelligence.passes.agentic.asyncio.sleep", new=AsyncMock()):
        with mock.patch(
            "sift.intelligence.passes.agentic.acompletion",
            new=AsyncMock(side_effect=[resp_err, resp_err, resp_err]),
        ):
            with mock.patch(
                "sift.intelligence.passes.agentic.generate_candidates",
                new=AsyncMock(return_value=[]),
            ) as mock_gen:
                findings = await agentic_review(file_input, plan, cap, {})

    mock_gen.assert_called_once()
    assert findings == []
