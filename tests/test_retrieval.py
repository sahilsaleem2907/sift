"""Tests for Phase 4 context retrieval."""
from src.intelligence.ast.function_extract import FunctionChunk
from src.intelligence.capability import ModelCapability
from src.intelligence.effort import EffortLevel, plan_for
from src.intelligence.retrieval import (
    FileContext,
    _callee_signatures,
    _semantic_before_after,
    build_context,
    trim_to_budget,
)


def _chunk(path: str, name: str, start: int, end: int, text: str) -> FunctionChunk:
    return FunctionChunk(
        path=path,
        name=name,
        start_line=start,
        end_line=end,
        text=text,
        content_hash="x",
    )


SIGNATURE_DIFF = """\
diff --git a/app/api.py b/app/api.py
--- a/app/api.py
+++ b/app/api.py
@@ -1,3 +1,3 @@
-def fetch(limit: int) -> list:
+def fetch(limit: int, offset: int = 0) -> list:
     return []
"""


def test_semantic_diff_extracts_old_body():
    mod_funcs = [
        _chunk(
            "app/api.py",
            "fetch",
            1,
            2,
            "def fetch(limit: int, offset: int = 0) -> list:\n    return []",
        )
    ]
    block = _semantic_before_after("app/api.py", SIGNATURE_DIFF, mod_funcs)
    assert "def fetch(limit: int) -> list" in block or "def fetch(limit: int)->" in block.replace(
        " ", ""
    )
    assert "offset" in block


def test_callee_resolution_finds_pr_functions():
    diff_a = """\
diff --git a/app/api.py b/app/api.py
--- a/app/api.py
+++ b/app/api.py
@@ -1,2 +1,3 @@
+    helper()
"""
    mod_funcs = {
        "app/api.py": [],
        "app/util.py": [
            _chunk("app/util.py", "helper", 1, 3, "def helper():\n    return 1"),
        ],
    }
    block = _callee_signatures("app/api.py", diff_a, mod_funcs)
    assert "app/util.py" in block
    assert "helper" in block


def test_trim_drops_lowest_priority_first():
    ctx = FileContext(
        diff="x" * 100,
        window_content="w" * 500,
        semantic_before_after="s" * 500,
        callee_signatures="c" * 500,
        caller_context="i" * 500,
        vector_snippets="v" * 500,
        static_tools="t" * 100,
    )
    trimmed = trim_to_budget(ctx, budget_chars=800)
    assert trimmed.diff == ctx.diff
    assert trimmed.vector_snippets == ""
    assert trimmed.callee_signatures == ""
    assert trimmed.semantic_before_after == ""


def test_trim_never_drops_diff():
    ctx = FileContext(
        diff="IMPORTANT_DIFF",
        window_content="w" * 10_000,
        semantic_before_after="s" * 10_000,
        vector_snippets="v" * 10_000,
    )
    trimmed = trim_to_budget(ctx, budget_chars=50)
    assert trimmed.diff == "IMPORTANT_DIFF"


def test_build_context_depth_zero_skips_semantic():
    plan = plan_for(EffortLevel.LOW)
    cap = ModelCapability(8192, 2048, False, False)
    ctx = build_context(
        "app/a.py",
        SIGNATURE_DIFF,
        {},
        plan,
        cap,
        path_to_content={"app/api.py": "def fetch():\n pass"},
        mod_funcs_by_path={},
        import_graph={},
    )
    assert ctx.semantic_before_after == ""
    assert ctx.callee_signatures == ""
