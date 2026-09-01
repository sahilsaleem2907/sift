"""Tests for the Java and Ruby linter paths in linter_runner.

Java regression coverage: single-file `javac` on a multi-module repo (e.g. keycloak) fails with
classpath-artifact errors (`cannot find symbol`) that are noise, not bugs. The Java linter must
(1) skip entirely when SpotBugs is not installed, and (2) never surface javac compile errors as
findings even when a compile fails.

Ruby regression coverage: rubocop lints diff-only single files with no repo Ruby-version signal,
so it defaults to a 2.7 parser that flags valid modern Ruby (endless methods, `it` param) as
fatal Lint/Syntax errors. _run_rubocop must write a controlled .rubocop.yml pinning a modern
TargetRubyVersion so those false syntax errors never appear.
"""
import shutil
import subprocess
import types

import pytest

from sift.core import linter_runner
from sift.core.linter_runner import (
    _ensure_rubocop_config,
    _run_rubocop,
    _run_spotbugs,
)


def test_spotbugs_skipped_when_not_installed(tmp_path, monkeypatch):
    # SpotBugs absent -> return [] without ever invoking javac.
    monkeypatch.setattr(linter_runner.shutil, "which", lambda name: None)

    def _boom(*a, **k):
        raise AssertionError("subprocess.run must not be called when SpotBugs is absent")

    monkeypatch.setattr(linter_runner.subprocess, "run", _boom)
    (tmp_path / "Foo.java").write_text("class Foo {}", encoding="utf-8")
    assert _run_spotbugs(tmp_path, "Foo.java", {"Foo.java": "class Foo {}"}) == []


def test_javac_compile_failure_yields_no_findings(tmp_path, monkeypatch):
    # SpotBugs present, but the single-file compile fails: the old javac-stderr fallback
    # would emit "cannot find symbol" as an ERROR finding. It must now yield nothing.
    monkeypatch.setattr(linter_runner.shutil, "which", lambda name: "/usr/local/bin/spotbugs")

    def _fake_javac(cmd, *a, **k):
        return types.SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Foo.java:5: error: cannot find symbol\n  symbol: class Bar\n",
        )

    monkeypatch.setattr(linter_runner.subprocess, "run", _fake_javac)
    (tmp_path / "Foo.java").write_text("class Foo { Bar b; }", encoding="utf-8")
    assert _run_spotbugs(tmp_path, "Foo.java", {"Foo.java": "class Foo { Bar b; }"}) == []


def test_ensure_rubocop_config_writes_modern_target(tmp_path):
    # First call writes .rubocop.yml pinning a modern parser + bug-only cops.
    cfg = _ensure_rubocop_config(tmp_path)
    assert cfg == tmp_path / ".rubocop.yml"
    content = cfg.read_text(encoding="utf-8")
    assert "TargetRubyVersion: 3.4" in content
    assert "DisabledByDefault: true" in content


def test_ensure_rubocop_config_idempotent(tmp_path):
    # Second call must not overwrite (rewrite would clobber a hand-authored config if any).
    cfg = _ensure_rubocop_config(tmp_path)
    cfg.write_text("sentinel", encoding="utf-8")
    again = _ensure_rubocop_config(tmp_path)
    assert again.read_text(encoding="utf-8") == "sentinel"


@pytest.mark.skipif(shutil.which("rubocop") is None, reason="rubocop not installed")
def test_rubocop_no_false_syntax_error_on_modern_ruby(tmp_path):
    # Endless method + `it` block param are valid Ruby 3.x. With the 2.7 default parser
    # rubocop reports these as fatal Lint/Syntax; the pinned TargetRubyVersion fixes it.
    src = (
        "# frozen_string_literal: true\n"
        "class Greeter\n"
        '  def greet = "hi"\n'
        "  def maybe(x) = x&.then { it.upcase }\n"
        "end\n"
    )
    (tmp_path / "greeter.rb").write_text(src, encoding="utf-8")
    issues = _run_rubocop(tmp_path, "greeter.rb")
    assert not any(i.get("rule_id") == "Lint/Syntax" for i in issues), issues
