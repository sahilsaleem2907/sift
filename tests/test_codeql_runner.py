"""Tests for CodeQL command construction.

Regression coverage for two production bugs found on the cal.com/keycloak runs:
- analyze passed the bare suite name `security-extended`, which CodeQL 2.24.2 cannot
  resolve → zero findings for every run. The fix uses the language-qualified suite path
  plus --additional-packs (the packs are not on the CLI's default resolve path).
- Java DB create used the default autobuild mode, which fails with no mvn/gradle. The
  fix passes --build-mode=none for languages that need+support it (java/csharp), gated so
  Go/C++/Swift (which don't support none) are untouched.
"""
from pathlib import Path

from sift.core.codeql_runner import (
    _build_analyze_cmd,
    _build_create_cmd,
    _should_use_build_mode_none,
)

SRC = Path("/tmp/src")
DB = Path("/tmp/db")
PACKS = "/home/u/.codeql/packages"


# ---- _build_create_cmd / build-mode gating ----

def test_create_java_uses_build_mode_none():
    cmd = _build_create_cmd(DB, ["java"], SRC)
    assert "--build-mode=none" in cmd
    assert "--language" in cmd and "java" in cmd


def test_create_csharp_uses_build_mode_none():
    assert "--build-mode=none" in _build_create_cmd(DB, ["csharp"], SRC)


def test_create_go_omits_build_mode_none():
    # Go does not support build-mode=none — passing it would error the create.
    assert "--build-mode=none" not in _build_create_cmd(DB, ["go"], SRC)


def test_create_python_omits_build_mode_none():
    # Python is already source-only; it doesn't need (or get) the flag.
    assert "--build-mode=none" not in _build_create_cmd(DB, ["python"], SRC)


def test_create_cluster_all_none_supported_gets_flag():
    cmd = _build_create_cmd(DB, ["java", "javascript"], SRC)
    assert "--db-cluster" in cmd
    assert "--build-mode=none" in cmd


def test_create_cluster_with_go_omits_flag():
    # java+go: a single --build-mode=none would break the go DB, so omit it.
    cmd = _build_create_cmd(DB, ["java", "go"], SRC)
    assert "--db-cluster" in cmd
    assert "--build-mode=none" not in cmd


def test_should_use_build_mode_none_matrix():
    assert _should_use_build_mode_none(["java"]) is True
    assert _should_use_build_mode_none(["csharp"]) is True
    assert _should_use_build_mode_none(["java", "javascript"]) is True
    assert _should_use_build_mode_none(["go"]) is False
    assert _should_use_build_mode_none(["python"]) is False   # supports none but doesn't need it
    assert _should_use_build_mode_none(["java", "go"]) is False


# ---- _build_analyze_cmd / suite resolution ----

def test_analyze_uses_language_qualified_suite_and_packs(tmp_path):
    packs = str(tmp_path)  # a real dir so --additional-packs is emitted
    cmd = _build_analyze_cmd(DB, "java", "security-extended", Path("/tmp/o.sarif"), packs)
    # The exact form proven to run (exit 0, 120 rules) in the CodeQL CLI.
    assert "codeql/java-queries:codeql-suites/java-security-extended.qls" in cmd
    assert cmd[cmd.index("--additional-packs") + 1] == packs
    # Never the broken bare suite name, never --download.
    assert "security-extended" not in cmd  # only the qualified .qls path, not the bare name
    assert "--download" not in cmd


def test_analyze_per_language_suite_name():
    cmd = _build_analyze_cmd(DB, "javascript", "security-extended", Path("/tmp/o.sarif"), PACKS)
    assert "codeql/javascript-queries:codeql-suites/javascript-security-extended.qls" in cmd


def test_analyze_default_suite_maps_to_code_scanning():
    cmd = _build_analyze_cmd(DB, "java", "default", Path("/tmp/o.sarif"), PACKS)
    assert "codeql/java-queries:codeql-suites/java-code-scanning.qls" in cmd


def test_analyze_security_and_quality_suite():
    cmd = _build_analyze_cmd(DB, "python", "security-and-quality", Path("/tmp/o.sarif"), PACKS)
    assert "codeql/python-queries:codeql-suites/python-security-and-quality.qls" in cmd


def test_analyze_omits_additional_packs_when_dir_missing():
    cmd = _build_analyze_cmd(DB, "java", "security-extended", Path("/tmp/o.sarif"), "/nonexistent/pack/dir")
    assert "--additional-packs" not in cmd
    # Suite is still qualified.
    assert "codeql/java-queries:codeql-suites/java-security-extended.qls" in cmd
