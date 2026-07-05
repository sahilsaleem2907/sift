"""Tests for src.core.pyright_runner."""
import json
from pathlib import Path
from unittest import mock

from src.core import pyright_runner
from src.core.pyright_runner import (
    _min_version_from_spec,
    detect_target_python,
    run_pyright,
)


# --- version-spec parsing (minimum of the supported range) --------------------

def test_min_version_from_range():
    assert _min_version_from_spec(">=3.11,<3.14") == "3.11"
    assert _min_version_from_spec(">=3.9") == "3.9"
    assert _min_version_from_spec("~=3.10") == "3.10"
    assert _min_version_from_spec("") is None


def test_detect_target_python_from_requires_python(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nrequires-python = ">=3.11,<3.14"\n', encoding="utf-8"
    )
    assert detect_target_python(tmp_path) == "3.11"


def test_detect_target_python_from_tool_pyright(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pyright]\npythonVersion = "3.13"\n', encoding="utf-8"
    )
    assert detect_target_python(tmp_path) == "3.13"


def test_detect_target_python_from_setup_cfg(tmp_path: Path):
    (tmp_path / "setup.cfg").write_text(
        "[options]\npython_requires = >=3.10\n", encoding="utf-8"
    )
    assert detect_target_python(tmp_path) == "3.10"


def test_detect_target_python_none_when_undeclared(tmp_path: Path):
    assert detect_target_python(tmp_path) is None


# --- output parsing: allowlist + drop noise + 0-based -> 1-based line ----------

_SAMPLE = {
    "generalDiagnostics": [
        {
            "file": "{root}/pkg/mod.py",
            "severity": "error",
            "message": '"shutdown" is not a known attribute of "Queue"',
            "rule": "reportAttributeAccessIssue",
            "range": {"start": {"line": 41, "character": 6}},  # 0-based -> line 42
        },
        {
            "file": "{root}/pkg/mod.py",
            "severity": "error",
            "message": 'Import "boto3" could not be resolved',
            "rule": "reportMissingImports",  # must be DROPPED
            "range": {"start": {"line": 0, "character": 0}},
        },
        {
            "file": "{root}/pkg/other.py",
            "severity": "error",
            "message": "No overloads for ... match",
            "rule": "reportCallIssue",
            "range": {"start": {"line": 9, "character": 0}},  # -> line 10
        },
    ]
}


def test_run_pyright_allowlist_and_line_mapping(tmp_path: Path):
    payload = json.dumps(_SAMPLE).replace("{root}", str(tmp_path))
    completed = mock.Mock(stdout=payload, stderr="", returncode=1)
    with mock.patch.object(pyright_runner, "_pyright_available", return_value=True):
        with mock.patch("subprocess.run", return_value=completed):
            out = run_pyright(tmp_path, ["pkg/mod.py", "pkg/other.py"], timeout=60)

    assert set(out.keys()) == {"pkg/mod.py", "pkg/other.py"}
    # reportMissingImports dropped -> only the attribute finding survives on mod.py
    assert len(out["pkg/mod.py"]) == 1
    f = out["pkg/mod.py"][0]
    assert f["line"] == 42                         # 0-based 41 -> 1-based 42
    assert f["severity"] == "ERROR"                # promoted as ERROR
    assert f["check_id"] == "pyright/reportAttributeAccessIssue"
    assert out["pkg/other.py"][0]["line"] == 10


def test_run_pyright_skips_when_unavailable(tmp_path: Path):
    with mock.patch.object(pyright_runner, "_pyright_available", return_value=False):
        assert run_pyright(tmp_path, ["x.py"], timeout=60) == {}


def test_run_pyright_pins_version_when_no_repo_config(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.11"\n', encoding="utf-8"
    )
    completed = mock.Mock(stdout='{"generalDiagnostics": []}', stderr="", returncode=0)
    with mock.patch.object(pyright_runner, "_pyright_available", return_value=True):
        with mock.patch("subprocess.run", return_value=completed) as run:
            run_pyright(tmp_path, ["a.py"], timeout=60)
    cmd = run.call_args[0][0]
    assert "--pythonversion" in cmd and "3.11" in cmd


def test_run_pyright_respects_repo_config(tmp_path: Path):
    (tmp_path / "pyrightconfig.json").write_text('{"pythonVersion": "3.13"}', encoding="utf-8")
    completed = mock.Mock(stdout='{"generalDiagnostics": []}', stderr="", returncode=0)
    with mock.patch.object(pyright_runner, "_pyright_available", return_value=True):
        with mock.patch("subprocess.run", return_value=completed) as run:
            run_pyright(tmp_path, ["a.py"], timeout=60)
    cmd = run.call_args[0][0]
    # repo config present -> we do NOT force a version (let their config win)
    assert "--pythonversion" not in cmd
