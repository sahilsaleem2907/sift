"""Tests for holistic PR pass and digest builder."""
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from sift.core.import_analyzer import CallerInfo
from sift.intelligence.ast.function_extract import FunctionChunk
from sift.intelligence.capability import ModelCapability
from sift.intelligence.effort import EffortLevel, plan_for
from sift.intelligence.passes.holistic import (
    PRDigest,
    _collect_external_callers,
    _parse_holistic_response,
    build_digest,
    review_holistic,
)
from sift.intelligence.passes.pipeline import PRMeta, run_pipeline_holistic
from sift.intelligence.schema import Certainty, Finding, Impact


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
        changed_functions=[
            {"path": "auth.py", "name": "verify", "lines": "1-3"},
            {"path": "app/api.py", "name": "handler", "lines": "4-8"},
        ],
        import_edges=[{"importer": "api.py", "imports_from": "auth.py", "symbols": ["verify"]}],
        per_file_findings=[],
    )
    mock_raw = """[{
        "path": "app/api.py",
        "line": 4,
        "def_location": "auth.py:2",
        "caller_location": "app/api.py:4",
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
        "sift.intelligence.passes.holistic._call_llm",
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
        "sift.intelligence.passes.holistic._call_llm",
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
        "sift.intelligence.passes.holistic._call_llm",
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
        "def_location": "app/auth.py:3",
        "caller_location": "app/api.py:4",
        "title": "Duplicate",
        "body": "same issue",
        "impact": "high",
        "certainty": "confirmed",
        "category": "correctness",
        "post_inline": true
    }, {
        "path": "app/other.py",
        "line": 2,
        "def_location": "app/auth.py:3",
        "caller_location": "app/other.py:2",
        "title": "New cross-file caller not updated",
        "body": "stale caller of changed verify() signature",
        "impact": "high",
        "certainty": "likely",
        "category": "correctness",
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
        # A caller_location outside the PR's changed files must now be find_callers-verified
        # (see _collect_external_callers) — "app/other.py:2" below is only kept because the
        # mocked find_callers legitimately reports it as a real usage of the changed symbol.
        repo_root="/fake/repo",
    )
    plan = plan_for(EffortLevel.BALANCED)
    cap = ModelCapability(8192, 2048, False, False)
    with mock.patch(
        "sift.core.code_intel.find_callers",
        return_value="app/other.py:2:  verify(token)",
    ), mock.patch(
        "sift.intelligence.passes.holistic._call_llm",
        new=AsyncMock(return_value=holistic_raw),
    ):
        findings = await run_pipeline_holistic([per_file], pr_meta, plan, cap)
    api_line4 = [f for f in findings if f.path == "app/api.py" and f.line == 4]
    assert len(api_line4) == 1
    assert api_line4[0].origin == "llm"
    other = [f for f in findings if f.path == "app/other.py" and f.origin == "holistic"]
    assert len(other) == 1
    assert other[0].post_inline is False  # unchanged file — no diff line to anchor inline


# ---- unchanged-caller contract drift (find_callers-verified) ----

def test_collect_external_callers_finds_usage_outside_changed_files():
    pr_meta = PRMeta(
        title="t", body="", repo_root="/fake/repo",
        mod_funcs_by_path={"app/auth.py": [_chunk("app/auth.py", "verify", 1, 5)]},
    )
    changed_functions = [{"path": "app/auth.py", "name": "verify", "lines": "1-5"}]
    changed_paths = frozenset({"app/auth.py"})
    with mock.patch(
        "sift.core.code_intel.find_callers",
        return_value="app/legacy_client.py:42:  if not verify(token): raise Error()",
    ):
        external = _collect_external_callers(pr_meta, changed_functions, changed_paths)
    assert len(external) == 1
    assert external[0] == {
        "symbol": "verify", "path": "app/legacy_client.py", "line": 42,
        "snippet": "if not verify(token): raise Error()",
    }


def test_collect_external_callers_excludes_changed_file_matches():
    # find_callers also matches usages inside files the PR already changed — those are already
    # visible to the model via the normal diff, so they must not be duplicated as "external".
    pr_meta = PRMeta(
        title="t", body="", repo_root="/fake/repo",
        mod_funcs_by_path={"app/auth.py": [_chunk("app/auth.py", "verify", 1, 5)]},
    )
    changed_functions = [{"path": "app/auth.py", "name": "verify", "lines": "1-5"}]
    changed_paths = frozenset({"app/auth.py", "app/api.py"})
    with mock.patch(
        "sift.core.code_intel.find_callers",
        return_value="app/api.py:9:  verify(token)",
    ):
        external = _collect_external_callers(pr_meta, changed_functions, changed_paths)
    assert external == []


def test_collect_external_callers_without_repo_root_returns_nothing():
    pr_meta = PRMeta(title="t", body="")  # no repo_root
    changed_functions = [{"path": "app/auth.py", "name": "verify", "lines": "1-5"}]
    assert _collect_external_callers(pr_meta, changed_functions, frozenset({"app/auth.py"})) == []


def test_parse_holistic_response_keeps_verified_external_caller():
    digest = PRDigest(
        title="t", body="", changed_functions=[], import_edges=[], per_file_findings=[],
        changed_paths=frozenset({"app/auth.py"}),
        external_callers=[{"symbol": "verify", "path": "app/legacy_client.py", "line": 42, "snippet": "verify(token)"}],
    )
    raw = """[{
        "path": "app/legacy_client.py",
        "line": 42,
        "def_location": "app/auth.py:3",
        "caller_location": "app/legacy_client.py:42",
        "title": "Stale caller of changed verify() contract",
        "body": "legacy client still assumes the old signature",
        "impact": "high",
        "certainty": "likely",
        "category": "correctness",
        "post_inline": true
    }]"""
    findings = _parse_holistic_response(raw, digest)
    assert len(findings) == 1
    assert findings[0].path == "app/legacy_client.py"
    assert findings[0].post_inline is False  # forced — unchanged file, no diff line to anchor to


def test_parse_holistic_response_drops_unverified_external_caller():
    digest = PRDigest(
        title="t", body="", changed_functions=[], import_edges=[], per_file_findings=[],
        changed_paths=frozenset({"app/auth.py"}),
        external_callers=[],  # nothing verified
    )
    raw = """[{
        "path": "app/some_random_file.py",
        "line": 10,
        "def_location": "app/auth.py:3",
        "caller_location": "app/some_random_file.py:10",
        "title": "Unverified claim",
        "body": "model asserts this without any tool evidence",
        "impact": "high",
        "certainty": "likely",
        "category": "correctness",
        "post_inline": true
    }]"""
    assert _parse_holistic_response(raw, digest) == []


def test_parse_holistic_response_drops_def_location_outside_changed_files():
    # Only a definition THIS PR actually changed counts as drift — a def_location claim
    # pointing outside the changed files is never legitimate, verified or not.
    digest = PRDigest(
        title="t", body="", changed_functions=[], import_edges=[], per_file_findings=[],
        changed_paths=frozenset({"app/api.py"}),
        external_callers=[{"symbol": "verify", "path": "app/auth.py", "line": 3, "snippet": "def verify(token):"}],
    )
    raw = """[{
        "path": "app/api.py",
        "line": 4,
        "def_location": "app/auth.py:3",
        "caller_location": "app/api.py:4",
        "title": "Def not actually changed by this PR",
        "body": "auth.py wasn't touched by this PR",
        "impact": "high",
        "certainty": "likely",
        "category": "correctness",
        "post_inline": true
    }]"""
    assert _parse_holistic_response(raw, digest) == []
