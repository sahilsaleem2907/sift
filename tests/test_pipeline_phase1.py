"""Behavioral contract: pipeline output matches review_file passthrough."""
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from src.intelligence.capability import ModelCapability
from src.intelligence.effort import EffortLevel, plan_for
from src.intelligence.passes.pipeline import FileReviewInput, PRMeta, run_pipeline


@pytest.mark.asyncio
async def test_pipeline_matches_review_file():
    diff = (
        "diff --git a/app/test.py b/app/test.py\n"
        "--- a/app/test.py\n"
        "+++ b/app/test.py\n"
        "@@ -1,2 +1,3 @@\n"
        " x = 1\n"
        "+y = None\n"
        "+print(y.name)\n"
    )
    path = "app/test.py"
    pr_ctx = {"title": "test", "body": ""}

    mock_comments = [
        {
            "line": 3,
            "body": "![BUG](https://img.shields.io/badge/BUG-AA0000) Null deref\n\ny may be None.",
            "post_inline": True,
        }
    ]

    with mock.patch(
        "src.intelligence.passes.candidates.llm_client.review_file",
        new=AsyncMock(return_value=mock_comments),
    ):
        from src.intelligence import llm_client

        old_comments = await llm_client.review_file(diff, path, pr_ctx)

        plan = plan_for(EffortLevel.LOW)
        cap = ModelCapability(8192, 2048, False, False)
        findings = await run_pipeline(
            [FileReviewInput(path, diff, pr_ctx)],
            PRMeta("test", ""),
            plan,
            cap,
        )

    pipeline_dicts = [f.to_comment_dict() for f in findings]
    expected = [
        {
            "path": path,
            "line": c["line"],
            "body": c["body"],
            "post_inline": c.get("post_inline", True),
        }
        for c in old_comments
    ]
    assert pipeline_dicts == expected
