"""Tests for Phase 4 agentic review loop."""
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import config
from src.intelligence.ast.function_extract import FunctionChunk
from src.intelligence.capability import ModelCapability
from src.intelligence.effort import EffortLevel, plan_for
from src.intelligence.passes.agentic import TOOLS, agentic_review
from src.intelligence.passes.pipeline import FileReviewInput


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
        "src.intelligence.passes.agentic.acompletion",
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
    final_msg = _mock_message(content="[]")

    resp_tool = MagicMock()
    resp_tool.choices = [MagicMock(message=always_tool)]
    resp_final = MagicMock()
    resp_final.choices = [MagicMock(message=final_msg)]

    old_max = config.SIFT_AGENTIC_MAX_STEPS
    config.SIFT_AGENTIC_MAX_STEPS = 2
    try:
        with mock.patch(
            "src.intelligence.passes.agentic.acompletion",
            new=AsyncMock(side_effect=[resp_tool, resp_tool, resp_final]),
        ) as mock_llm:
            findings = await agentic_review(file_input, plan, cap, {"app/a.py": "x=1\n"})
            assert mock_llm.call_count == 3
            assert findings == []
    finally:
        config.SIFT_AGENTIC_MAX_STEPS = old_max


@pytest.mark.asyncio
async def test_agentic_non_tool_model_falls_back_to_generate_candidates():
    plan = plan_for(EffortLevel.HIGH)
    cap = ModelCapability(8192, 2048, False, False)
    file_input = FileReviewInput(path="app/a.py", file_diff="+x\n", pr_context={})

    with mock.patch(
        "src.intelligence.passes.agentic.agentic_review",
        new=AsyncMock(),
    ) as mock_agentic:
        from src.intelligence.passes.pipeline import run_pipeline_per_file

        with mock.patch(
            "src.intelligence.passes.candidates.llm_client.review_file",
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
        "src.intelligence.passes.agentic.acompletion",
        new=AsyncMock(side_effect=RuntimeError("api down")),
    ):
        with mock.patch(
            "src.intelligence.passes.agentic.generate_candidates",
            new=AsyncMock(return_value=[]),
        ) as mock_gen:
            findings = await agentic_review(file_input, plan, cap, {})
            mock_gen.assert_called_once()
            assert findings == []
