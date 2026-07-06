"""Tests for repo-wide code-intelligence fact-tool backends."""
import subprocess
import textwrap
from pathlib import Path

import pytest

from src.core import code_intel


@pytest.fixture()
def repo(tmp_path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "base.py").write_text(textwrap.dedent("""
        from abc import ABC, abstractmethod

        class Handler(ABC):
            @abstractmethod
            def do_a(self): ...
            @abstractmethod
            def do_b(self): ...
            def helper(self): return 1
    """))
    (tmp_path / "pkg" / "child.py").write_text(textwrap.dedent("""
        from pkg.base import Handler

        class GoodChild(Handler):
            def do_a(self): return 1
            def do_b(self): return 2

        class BadChild(Handler):
            pass
    """))
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
        cwd=tmp_path, check=True,
    )
    return str(tmp_path)


def test_read_file(repo):
    out = code_intel.read_file(repo, "pkg/base.py", 1, 3)
    assert "abstractmethod" in out or "Handler" in out


def test_read_file_escape_guard(repo):
    assert code_intel.read_file(repo, "../../etc/passwd").startswith("[not found")


def test_search_repo(repo):
    out = code_intel.search_repo(repo, "abstractmethod")
    assert "base.py" in out


def test_find_definition(repo):
    out = code_intel.find_definition(repo, "Handler")
    assert "base.py" in out and "class Handler" in out


def test_find_callers_none(repo):
    assert code_intel.find_callers(repo, "do_a").startswith(("pkg/", "[no callers"))


def test_get_mro_flags_unimplemented(repo):
    out = code_intel.get_mro(repo, "pkg/child.py", "BadChild")
    assert "unimplemented" in out
    assert "do_a" in out and "do_b" in out
    assert "TypeError" in out


def test_get_mro_clean_child(repo):
    out = code_intel.get_mro(repo, "pkg/child.py", "GoodChild")
    assert "unimplemented_by_GoodChild=(none)" in out
    assert "does not implement abstract method" not in out


def test_missing_file(repo):
    assert code_intel.get_mro(repo, "pkg/nope.py", "X").startswith("[not found")
