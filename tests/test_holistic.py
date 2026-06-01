"""Tests for holistic PR pass and digest builder."""
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from src.core.import_analyzer import CallerInfo
from src.intelligence.ast.function_extract import FunctionChunk
from src.intelligence.capability import ModelCapability
from src.intelligence.effort import EffortLevel, plan_for
from src.intelligence.passes.holistic import (
    PRDigest,
    build_digest,
    review_holistic,
)
from src.intelligence.passes.pipeline import PRMeta, run_pipeline_holistic
from src.intelligence.schema import Certainty, Finding, Impact


def _chunk(path: str, name: str, start: int, end: int) -> FunctionChunk:
    return FunctionChunk(
        path=path,
        name=name,
        start_line=start,
        end_line=end,
        text=f"def {name}(): pass",
        content_hash="abc",
    )


def test_build_digest_extracts_functions():
    pr_meta = PRMeta(
        title="t",
        body="",
        mod_funcs_by_path={
            "a.py": [_chunk("a.py", "foo", 1, 10)],
            "b.py": [_chunk("b.py", "bar", 5, 18)],
        },
    )
    digest = build_digest(pr_meta, [])
    assert len(digest.changed_functions) == 2
    paths = {cf["path"] for cf in digest.changed_functions}
    assert paths == {"a.py", "b.py"}


def test_build_digest_extracts_import_edges():
    pr_meta = PRMeta(
        title="t",
        body="",
        import_graph={
            "api.py": [CallerInfo(changed_path="auth.py", function_names=("verify",))],
        },
    )
    digest = build_digest(pr_meta, [])
    assert len(digest.import_edges) == 1
    assert digest.import_edges[0]["importer"] == "api.py"
    assert digest.import_edges[0]["imports_from"] == "auth.py"
    assert digest.import_edges[0]["symbols"] == ["verify"]


@pytest.mark.asyncio
async def test_review_holistic_returns_findings():
    digest = PRDigest(
        title="sig change",
        body="",
        changed_functions=[{"path": "auth.py", "name": "verify", "lines": "1-3"}],
        import_edges=[{"importer": "api.py", "imports_from": "auth.py", "symbols": ["verify"]}],
        per_file_findings=[],
    )
    mock_raw = """[{
        "path": "app/api.py",
        "line": 4,
        "title": "Caller not updated",
        "body": "verify_token signature changed",
        "impact": "high",
        "certainty": "confirmed",
        "category": "correctness",
        "post_inline": true
    }]"""
    plan = plan_for(EffortLevel.BALANCED)
    cap = ModelCapability(8192, 2048, False, False)
    with mock.patch(
        "src.intelligence.passes.holistic._call_llm",
        new=AsyncMock(return_value=mock_raw),
    ):
        findings = await review_holistic(digest, plan, cap)
    assert len(findings) == 1
    assert findings[0].origin == "holistic"
    assert findings[0].path == "app/api.py"
    assert findings[0].line == 4


@pytest.mark.asyncio
async def test_review_holistic_empty_when_no_edges():
    digest = PRDigest(
        title="single file",
        body="",
        changed_functions=[{"path": "only.py", "name": "fn", "lines": "1-2"}],
        import_edges=[],
        per_file_findings=[],
    )
    plan = plan_for(EffortLevel.BALANCED)
    cap = ModelCapability(8192, 2048, False, False)
    with mock.patch(
        "src.intelligence.passes.holistic._call_llm",
        new=AsyncMock(),
    ) as mock_llm:
        findings = await review_holistic(digest, plan, cap)
    assert findings == []
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_review_holistic_parse_failure_returns_empty():
    digest = PRDigest(
        title="multi",
        body="",
        changed_functions=[
            {"path": "a.py", "name": "f1", "lines": "1-2"},
            {"path": "b.py", "name": "f2", "lines": "3-4"},
        ],
        import_edges=[{"importer": "b.py", "imports_from": "a.py", "symbols": []}],
        per_file_findings=[],
    )
    plan = plan_for(EffortLevel.BALANCED)
    cap = ModelCapability(8192, 2048, False, False)
    with mock.patch(
        "src.intelligence.passes.holistic._call_llm",
        new=AsyncMock(return_value="not valid json at all"),
    ):
        findings = await review_holistic(digest, plan, cap)
    assert findings == []


@pytest.mark.asyncio
async def test_pipeline_dedupes_holistic_against_per_file():
    per_file = Finding(
        path="app/api.py",
        line=4,
        title="Existing",
        body="![WARNING](https://img.shields.io/badge/WARNING-B8860B) Existing",
        impact=Impact.MEDIUM,
        certainty=Certainty.LIKELY,
        category="correctness",
        origin="llm",
    )
    holistic_raw = """[{
        "path": "app/api.py",
        "line": 4,
        "title": "Duplicate",
        "body": "same issue",
        "impact": "high",
        "certainty": "confirmed",
        "category": "correctness",
        "post_inline": true
    }, {
        "path": "app/other.py",
        "line": 2,
        "title": "New cross-file",
        "body": "design issue",
        "impact": "low",
        "certainty": "likely",
        "category": "design",
        "post_inline": true
    }]"""
    pr_meta = PRMeta(
        title="PR",
        body="",
        import_graph={"app/api.py": [CallerInfo("app/auth.py", ("verify",))]},
        mod_funcs_by_path={
            "app/api.py": [_chunk("app/api.py", "handle", 1, 10)],
            "app/auth.py": [_chunk("app/auth.py", "verify", 1, 5)],
        },
    )
    plan = plan_for(EffortLevel.BALANCED)
    cap = ModelCapability(8192, 2048, False, False)
    with mock.patch(
        "src.intelligence.passes.holistic._call_llm",
        new=AsyncMock(return_value=holistic_raw),
    ):
        findings = await run_pipeline_holistic([per_file], pr_meta, plan, cap)
    api_line4 = [f for f in findings if f.path == "app/api.py" and f.line == 4]
    assert len(api_line4) == 1
    assert api_line4[0].origin == "llm"
    assert any(f.path == "app/other.py" and f.origin == "holistic" for f in findings)
