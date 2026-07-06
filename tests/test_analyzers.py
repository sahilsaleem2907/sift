"""Tests for the deterministic AST verdict analyzers (language seam)."""
from src.core.analyzers import run_analyzers


def _full_add_diff(src: str) -> str:
    """A diff that marks every line of `src` as newly added."""
    lines = src.splitlines()
    header = f"@@ -0,0 +1,{len(lines)} @@\n"
    return header + "".join(f"+{ln}\n" for ln in lines)


_DATACLASS_SRC = """from dataclasses import dataclass, field


@dataclass
class Job:
    name: str
    tags: list = []
    meta: dict = field(default_factory=dict)
    counts: dict = {}
    seen: set = set()
"""


def test_flags_mutable_dataclass_defaults():
    out = run_analyzers("job.py", _DATACLASS_SRC, _full_add_diff(_DATACLASS_SRC))
    lines = {f["line"] for f in out}
    # tags=[], counts={}, seen=set() are mutable; meta=field(default_factory) is not.
    assert lines == {7, 9, 10}
    assert all(f["check_id"] == "analyzer/mutable-default" for f in out)
    assert all(f["severity"] == "ERROR" for f in out)


def test_ignores_default_factory():
    out = run_analyzers("job.py", _DATACLASS_SRC, _full_add_diff(_DATACLASS_SRC))
    # line 8 is meta = field(default_factory=dict) — correct form, never flagged.
    assert 8 not in {f["line"] for f in out}


def test_only_flags_changed_lines():
    # Diff touches an unrelated import; the mutable-default field is unchanged.
    diff = "@@ -1,2 +1,3 @@\n+import os\n"
    assert run_analyzers("job.py", _DATACLASS_SRC, diff) == []


def test_non_dataclass_not_flagged():
    src = "class Job:\n    tags: list = []\n"
    assert run_analyzers("job.py", src, _full_add_diff(src)) == []


def test_non_python_graceful():
    assert run_analyzers("main.go", "package main\n", "@@ -0,0 +1,1 @@\n+package main\n") == []


def test_empty_inputs():
    assert run_analyzers("job.py", "", "") == []
    assert run_analyzers("job.py", _DATACLASS_SRC, "") == []
