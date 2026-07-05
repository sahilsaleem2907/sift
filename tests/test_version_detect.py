"""Tests for src.core.version_detect."""
from src.core.version_detect import (
    CANDIDATE_FILES,
    GoVersionDetector,
    JavaVersionDetector,
    NodeTsVersionDetector,
    PythonVersionDetector,
    RubyVersionDetector,
    _min_version_from_spec,
    detect_targets,
    target_for_path,
)


def _reader(files: dict):
    return lambda name: files.get(name)


# --- spec parsing -------------------------------------------------------------

def test_min_version_from_spec():
    assert _min_version_from_spec(">=3.11,<3.14") == "3.11"
    assert _min_version_from_spec(">=3.9") == "3.9"
    assert _min_version_from_spec("") is None


# --- per-language detectors ---------------------------------------------------

def test_python_requires_min():
    r = _reader({"pyproject.toml": '[project]\nrequires-python = ">=3.11,<3.14"\n'})
    t = PythonVersionDetector().detect(r)
    assert t and t.language == "python" and "3.11" in t.summary


def test_python_tool_pyright_fallback():
    r = _reader({"pyproject.toml": '[tool.pyright]\npythonVersion = "3.13"\n'})
    t = PythonVersionDetector().detect(r)
    assert t and "3.13" in t.summary


def test_python_none_when_undeclared():
    assert PythonVersionDetector().detect(_reader({})) is None


def test_node_engines_and_tsconfig():
    r = _reader({
        "package.json": '{"engines": {"node": ">=20"}}',
        "tsconfig.json": '{"compilerOptions": {"target": "ES2022"}}',
    })
    t = NodeTsVersionDetector().detect(r)
    assert t and t.language == "typescript"
    assert "Node >=20" in t.summary and "ES2022" in t.summary


def test_go_mod():
    t = GoVersionDetector().detect(_reader({"go.mod": "module x\n\ngo 1.22\n"}))
    assert t and t.summary == "Go 1.22 (go.mod)"


def test_ruby_version_file():
    t = RubyVersionDetector().detect(_reader({".ruby-version": "3.3.1\n"}))
    assert t and "Ruby 3.3" in t.summary


def test_java_pom_release():
    t = JavaVersionDetector().detect(_reader({"pom.xml": "<maven.compiler.release>17</maven.compiler.release>"}))
    assert t and "Java 17" in t.summary


# --- dispatch -----------------------------------------------------------------

def test_detect_targets_multi_language():
    r = _reader({
        "pyproject.toml": '[project]\nrequires-python = ">=3.12"\n',
        "go.mod": "go 1.21\n",
    })
    targets = detect_targets(r)
    assert set(targets) == {"python", "go"}


def test_target_for_path_selects_by_extension():
    targets = detect_targets(_reader({"pyproject.toml": '[project]\nrequires-python = ">=3.13"\n'}))
    assert "3.13" in target_for_path("src/a.py", targets).summary
    assert target_for_path("src/a.go", targets) is None       # no go.mod -> no target
    assert target_for_path("README.md", targets) is None      # unknown language


def test_candidate_files_covers_all_detectors():
    for f in ("pyproject.toml", "package.json", "tsconfig.json", "go.mod", ".ruby-version", "pom.xml"):
        assert f in CANDIDATE_FILES
