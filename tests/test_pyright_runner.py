"""Tests for src.core.pyright_runner."""
import json
from pathlib import Path
from unittest import mock

from src.core import pyright_runner
from src.core.pyright_runner import detect_target_python, run_pyright


# --- target-version detection (delegates to the shared PythonVersionDetector) --

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
            "message": "Expected 1 positional argument",
            "rule": "reportCallIssue",  # reliable rule → kept
            "range": {"start": {"line": 41, "character": 6}},  # 0-based -> line 42
        },
        {
            "file": "{root}/pkg/mod.py",
            "severity": "error",
            "message": 'Import "boto3" could not be resolved',
            "rule": "reportMissingImports",  # must be DROPPED (not allowlisted)
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
    # reportMissingImports dropped -> only the reportCallIssue survives on mod.py
    assert len(out["pkg/mod.py"]) == 1
    f = out["pkg/mod.py"][0]
    assert f["line"] == 42                         # 0-based 41 -> 1-based 42
    assert f["severity"] == "ERROR"                # promoted as ERROR
    assert f["check_id"] == "pyright/reportCallIssue"
    assert out["pkg/other.py"][0]["line"] == 10


# --- reportAttributeAccessIssue curation (Option 4): only fire when provably absent ---

def _run_with_diag(tmp_path: Path, diag: dict, changed: list[str]) -> dict:
    payload = json.dumps({"generalDiagnostics": [diag]}).replace("{root}", str(tmp_path))
    completed = mock.Mock(stdout=payload, stderr="", returncode=1)
    with mock.patch.object(pyright_runner, "_pyright_available", return_value=True):
        with mock.patch("subprocess.run", return_value=completed):
            return run_pyright(tmp_path, changed, timeout=60)


def _import_symbol_diag(tmp_path: Path, symbol: str = "cache") -> dict:
    return {
        "file": f"{tmp_path}/caller.py",
        "severity": "error",
        "message": f'"{symbol}" is unknown import symbol',
        "rule": "reportAttributeAccessIssue",
        "range": {"start": {"line": 0, "character": 0}},  # line 1: the import
    }


def test_curate_drops_reexport_false_positive(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    # module re-exports `cache` from an (uninstalled) third-party dep → pyright FP
    (tmp_path / "pkg" / "thing.py").write_text("from django.core.cache import cache\n")
    (tmp_path / "caller.py").write_text("from pkg.thing import cache\nprint(cache)\n")
    out = _run_with_diag(tmp_path, _import_symbol_diag(tmp_path), ["caller.py"])
    assert out == {}  # symbol is bound in the module → dropped


def test_curate_keeps_genuinely_missing_symbol(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "thing.py").write_text("VALUE = 1\n")  # no `cache`
    (tmp_path / "caller.py").write_text("from pkg.thing import cache\n")
    out = _run_with_diag(tmp_path, _import_symbol_diag(tmp_path), ["caller.py"])
    assert out["caller.py"][0]["check_id"] == "pyright/reportAttributeAccessIssue"


def test_curate_drops_thirdparty_import_symbol(tmp_path: Path):
    # module not in the clone at all (third-party) → unverifiable → dropped
    (tmp_path / "caller.py").write_text("from django.core.cache import cache\n")
    out = _run_with_diag(tmp_path, _import_symbol_diag(tmp_path), ["caller.py"])
    assert out == {}


def test_curate_drops_attribute_on_type(tmp_path: Path):
    (tmp_path / "m.py").write_text("x = 1\n")
    diag = {
        "file": f"{tmp_path}/m.py",
        "severity": "error",
        "message": '"slug" is not a known attribute of "MethodType"',
        "rule": "reportAttributeAccessIssue",
        "range": {"start": {"line": 0, "character": 0}},
    }
    assert _run_with_diag(tmp_path, diag, ["m.py"]) == {}  # attribute-on-type → lifted (dropped from floor)


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


# --- src-layout: inject extraPaths via an ephemeral config so first-party imports resolve ---

def test_run_pyright_src_layout_injects_ephemeral_extrapaths(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.11"\n', encoding="utf-8"
    )
    (tmp_path / "src").mkdir()

    captured: dict = {}

    def _capture(cmd, *a, **k):
        cfg = tmp_path / "pyrightconfig.json"
        captured["existed_during_run"] = cfg.is_file()
        if cfg.is_file():
            captured["cfg"] = json.loads(cfg.read_text())
        return mock.Mock(stdout='{"generalDiagnostics": []}', stderr="", returncode=0)

    with mock.patch.object(pyright_runner, "_pyright_available", return_value=True):
        with mock.patch("subprocess.run", side_effect=_capture) as run:
            run_pyright(tmp_path, ["a.py"], timeout=60)

    # config existed while pyright ran, with src on the path + version pinned there
    assert captured["existed_during_run"] is True
    assert captured["cfg"]["extraPaths"] == ["src"]
    assert captured["cfg"]["pythonVersion"] == "3.11"
    assert captured["cfg"]["reportMissingImports"] is False
    # version pin goes through the config, not the CLI
    assert "--pythonversion" not in run.call_args[0][0]
    # ephemeral file is cleaned up afterward
    assert not (tmp_path / "pyrightconfig.json").exists()


def test_run_pyright_no_src_layout_keeps_cli_version(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.12"\n', encoding="utf-8"
    )
    # no src/ dir → no ephemeral config; falls back to --pythonversion
    completed = mock.Mock(stdout='{"generalDiagnostics": []}', stderr="", returncode=0)
    with mock.patch.object(pyright_runner, "_pyright_available", return_value=True):
        with mock.patch("subprocess.run", return_value=completed) as run:
            run_pyright(tmp_path, ["a.py"], timeout=60)
    cmd = run.call_args[0][0]
    assert "--pythonversion" in cmd and "3.12" in cmd
    assert not (tmp_path / "pyrightconfig.json").exists()


def test_run_pyright_src_layout_never_clobbers_real_config(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "pyrightconfig.json").write_text('{"pythonVersion": "3.13"}', encoding="utf-8")
    completed = mock.Mock(stdout='{"generalDiagnostics": []}', stderr="", returncode=0)
    with mock.patch.object(pyright_runner, "_pyright_available", return_value=True):
        with mock.patch("subprocess.run", return_value=completed):
            run_pyright(tmp_path, ["a.py"], timeout=60)
    # a real repo config is left exactly as-is (not overwritten, not deleted)
    assert json.loads((tmp_path / "pyrightconfig.json").read_text()) == {"pythonVersion": "3.13"}
