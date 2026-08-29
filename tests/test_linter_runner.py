"""Tests for the Java linter path in linter_runner.

Regression coverage: single-file `javac` on a multi-module repo (e.g. keycloak) fails with
classpath-artifact errors (`cannot find symbol`) that are noise, not bugs. The Java linter must
(1) skip entirely when SpotBugs is not installed, and (2) never surface javac compile errors as
findings even when a compile fails.
"""
import subprocess
import types

from sift.core import linter_runner
from sift.core.linter_runner import _run_spotbugs


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
