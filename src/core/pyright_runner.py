"""Pyright type-checker: deterministic API/version-existence findings.

Runs pyright against a full repo checkout (from repo_cache) pinned to the repo's
target Python version, and returns a narrow, high-precision subset of diagnostics
(attribute-existence / bad-call-signature) shaped like Semgrep findings so they can
flow through promote_static_findings as critic-exempt.

Design notes (see plan): bare clone, no deps installed — pyright resolves stdlib via
its bundled typeshed and self-suppresses on unresolved third-party imports. We respect
the repo's own pyright config when present (honours intent + their extraPaths/stubs);
otherwise we pin the version we detect from requires-python. Rule/line scoping is done
by post-filtering, so it only ever narrows.
"""
import json
import logging
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Only surface the API/existence class — high precision, targets the golden class.
# reportMissingImports & general type-strictness are intentionally excluded.
_ALLOWED_RULES = frozenset({"reportAttributeAccessIssue", "reportCallIssue"})


def _has_repo_pyright_config(repo_root: Path) -> bool:
    """True if the repo declares its own pyright config (respected as-is when present)."""
    if (repo_root / "pyrightconfig.json").is_file():
        return True
    pyproject = repo_root / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            if isinstance(data.get("tool"), dict) and "pyright" in data["tool"]:
                return True
        except Exception:
            pass
    return False


def _min_version_from_spec(spec: str) -> Optional[str]:
    """Extract the minimum 'X.Y' from a requires-python spec (e.g. '>=3.11,<3.14' -> '3.11')."""
    if not spec:
        return None
    m = re.search(r">=\s*(\d+)\.(\d+)", spec)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    m = re.search(r"~=\s*(\d+)\.(\d+)", spec) or re.search(r"==\s*(\d+)\.(\d+)", spec)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    m = re.search(r"(\d+)\.(\d+)", spec)  # last resort: first version-looking token
    return f"{m.group(1)}.{m.group(2)}" if m else None


def detect_target_python(repo_root: Path) -> Optional[str]:
    """Detect the repo's minimum supported Python 'X.Y', or None if undeterminable.

    A version-compat bug is a bug if it breaks on any *supported* version, so we use
    the MINIMUM of requires-python. Priority: pyproject requires-python -> setup.cfg /
    setup.py python_requires -> [tool.mypy]/[tool.pyright] version -> .python-version.
    """
    pyproject = repo_root / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            proj = data.get("project")
            if isinstance(proj, dict):
                v = _min_version_from_spec(str(proj.get("requires-python") or ""))
                if v:
                    return v
            tool = data.get("tool") if isinstance(data.get("tool"), dict) else {}
            pr = tool.get("pyright") if isinstance(tool.get("pyright"), dict) else {}
            if pr.get("pythonVersion"):
                return str(pr["pythonVersion"])
            mp = tool.get("mypy") if isinstance(tool.get("mypy"), dict) else {}
            if mp.get("python_version"):
                return str(mp["python_version"])
        except Exception as e:
            logger.debug("pyproject.toml parse failed for version detection: %s", e)

    for name in ("setup.cfg", "setup.py"):
        f = repo_root / name
        if f.is_file():
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:
                continue
            m = re.search(r"python_requires\s*=\s*['\"]?([^'\"\n]+)", text)
            if m:
                v = _min_version_from_spec(m.group(1))
                if v:
                    return v

    pv = repo_root / ".python-version"
    if pv.is_file():
        try:
            m = re.search(r"(\d+)\.(\d+)", pv.read_text(encoding="utf-8"))
            if m:
                return f"{m.group(1)}.{m.group(2)}"
        except Exception:
            pass
    return None


def _pyright_available() -> bool:
    try:
        subprocess.run(["pyright", "--version"], capture_output=True, text=True, timeout=15, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.debug("pyright not found in PATH; skipping pyright")
        return False


def run_pyright(
    repo_root: Path,
    changed_py_paths: List[str],
    timeout: int,
) -> Dict[str, List[dict]]:
    """Type-check the changed Python files; return allowlisted findings by repo-relative path.

    Returns Dict[path, list of {line, message, severity:"ERROR", check_id:"pyright/<rule>"}].
    Returns {} on any failure (not installed, timeout, config error, no JSON) — graceful skip.
    """
    if not changed_py_paths or not _pyright_available():
        return {}

    cmd = ["pyright", "--outputjson"]
    # Respect the repo's own pyright config when present (it carries their version,
    # ignores, extraPaths/stubs). Otherwise pin the version we detect.
    if not _has_repo_pyright_config(repo_root):
        target = detect_target_python(repo_root)
        if target:
            cmd += ["--pythonversion", target]
            logger.debug("[pyright] no repo config; pinning pythonVersion=%s", target)
    else:
        logger.debug("[pyright] respecting repo's own pyright config")
    cmd += list(changed_py_paths)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(repo_root),
        )
    except subprocess.TimeoutExpired:
        logger.warning("pyright timed out after %ss", timeout)
        return {}
    except Exception as e:
        logger.warning("pyright failed to run: %s", e)
        return {}

    # pyright exits non-zero when it finds errors; that's expected. Only bail if no JSON.
    if not result.stdout.strip():
        logger.debug("pyright produced no JSON output (stderr: %s)", result.stderr[:300])
        return {}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.warning("pyright JSON parse failed: %s", e)
        return {}

    by_path: Dict[str, List[dict]] = {}
    for d in data.get("generalDiagnostics") or []:
        rule = d.get("rule") or ""
        if rule not in _ALLOWED_RULES:
            continue
        file_abs = d.get("file") or ""
        try:
            rel = str(Path(file_abs).resolve().relative_to(repo_root.resolve())).replace("\\", "/")
        except (ValueError, OSError):
            rel = Path(file_abs).name
        # pyright range lines are 0-based; convert to 1-based new-file line numbers.
        start = (d.get("range") or {}).get("start") or {}
        line = int(start.get("line", 0)) + 1
        by_path.setdefault(rel, []).append({
            "line": line,
            "message": (d.get("message") or "").strip(),
            "severity": "ERROR",
            "check_id": f"pyright/{rule}",
        })
    return by_path
