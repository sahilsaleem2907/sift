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

from src.core.version_detect import PythonVersionDetector

logger = logging.getLogger(__name__)

# Only surface the API/existence class — high precision, targets the golden class.
# reportMissingImports & general type-strictness are intentionally excluded.
# reportAbstractUsage: instantiating a class with unimplemented abstract methods is
# a guaranteed TypeError at runtime — high precision, so it joins the floor.
_ALLOWED_RULES = frozenset(
    {"reportAttributeAccessIssue", "reportCallIssue", "reportAbstractUsage"}
)


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


def detect_target_python(repo_root: Path) -> Optional[str]:
    """Detect the repo's minimum supported Python 'X.Y', or None if undeterminable.

    Delegates to the shared PythonVersionDetector (single source of truth) using a
    clone-backed reader. Returns just the bare 'X.Y' string that pyright's
    --pythonversion flag expects (the detector's RuntimeTarget carries a prose summary).
    """
    def _read(name: str) -> Optional[str]:
        f = repo_root / name
        if not f.is_file():
            return None
        try:
            return f.read_text(encoding="utf-8")
        except Exception:
            return None

    target = PythonVersionDetector().detect(_read)
    if target is None:
        return None
    m = re.search(r"(\d+)\.(\d+)", target.summary)
    return f"{m.group(1)}.{m.group(2)}" if m else None


def _pyright_available() -> bool:
    try:
        subprocess.run(["pyright", "--version"], capture_output=True, text=True, timeout=15, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.debug("pyright not found in PATH; skipping pyright")
        return False


def _write_ephemeral_pyright_config(repo_root: Path, target: Optional[str]) -> Optional[Path]:
    """Write a throwaway pyrightconfig.json so a src-layout repo's first-party imports resolve.

    Without `src` on pyright's import path, `import <pkg>...` (living under src/) fails to resolve,
    and pyright reports valid first-party symbols as missing (reportAttributeAccessIssue false
    positives). Only called when the repo ships no pyright config, so we never clobber a real one.
    Returns the written path (to be removed by the caller), or None on skip/failure.
    """
    path = repo_root / "pyrightconfig.json"
    if path.exists():
        return None  # never overwrite an existing config (incl. a stale ephemeral one)
    cfg: dict = {"extraPaths": ["src"], "reportMissingImports": False}
    if target:
        cfg["pythonVersion"] = target
    try:
        path.write_text(json.dumps(cfg), encoding="utf-8")
        return path
    except OSError as e:
        logger.debug("[pyright] could not write ephemeral config: %s", e)
        return None


_UNKNOWN_IMPORT_RE = re.compile(r'"([^"]+)"\s+is unknown import symbol', re.IGNORECASE)


def _import_module_for_line(repo_root: Path, rel: str, line: int) -> Optional[str]:
    """Read the finding's source line (and a few above) to find the `from Y import …` module."""
    try:
        lines = (repo_root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if not (1 <= line <= len(lines)):
        return None
    # The diagnostic sits on the imported name; for multi-line imports scan a few lines up.
    for i in range(line - 1, max(-1, line - 6), -1):
        m = re.match(r"\s*from\s+([\w.]+)\s+import\b", lines[i])
        if m:
            return m.group(1)
    return None


def _resolve_module_file(repo_root: Path, dotted: str) -> Optional[Path]:
    """Resolve an absolute dotted module to a file under the clone roots (repo_root, repo_root/src)."""
    if not dotted or dotted.startswith("."):
        return None  # relative imports need package context we don't resolve here
    rel = dotted.replace(".", "/")
    for root in (repo_root, repo_root / "src"):
        for cand in (root / f"{rel}.py", root / rel / "__init__.py"):
            if cand.is_file():
                return cand
    return None


def _module_binds_symbol(module_file: Path, symbol: str) -> bool:
    """True if `symbol` is bound at module top level (def/class/assignment/import).

    Used to disprove a pyright "unknown import symbol" false positive: if the module
    actually binds the name (commonly a re-export pyright couldn't follow because the
    upstream third-party dep isn't installed), the finding is spurious.
    """
    try:
        src = module_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    s = re.escape(symbol)
    patterns = (
        rf"^\s*(?:async\s+)?(?:def|class)\s+{s}\b",  # def/class symbol
        rf"^\s*{s}\s*[:=]",                            # symbol = …  /  symbol: T = …
        rf"^\s*(?:from\s+[\w.]+\s+)?import\b.*\b{s}\b",  # import/from-import (incl. `as {s}`)
    )
    return any(re.search(p, src, re.MULTILINE) for p in patterns)


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
    ephemeral_config: Optional[Path] = None
    if not _has_repo_pyright_config(repo_root):
        target = detect_target_python(repo_root)
        if (repo_root / "src").is_dir():
            # src-layout repo with no pyright config: inject extraPaths=[src] (and the
            # version pin) via an ephemeral config so first-party imports resolve —
            # otherwise valid symbols are flagged as missing (false positives).
            ephemeral_config = _write_ephemeral_pyright_config(repo_root, target)
        if ephemeral_config is not None:
            logger.debug(
                "[pyright] src-layout, no repo config; ephemeral extraPaths=[src] pythonVersion=%s",
                target,
            )
        elif target:
            cmd += ["--pythonversion", target]
            logger.debug("[pyright] no repo config; pinning pythonVersion=%s", target)
    else:
        logger.debug("[pyright] respecting repo's own pyright config")
    cmd += list(changed_py_paths)

    try:
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
            message = (d.get("message") or "").strip()
            file_abs = d.get("file") or ""
            try:
                rel = str(Path(file_abs).resolve().relative_to(repo_root.resolve())).replace("\\", "/")
            except (ValueError, OSError):
                rel = Path(file_abs).name
            # pyright range lines are 0-based; convert to 1-based new-file line numbers.
            start = (d.get("range") or {}).get("start") or {}
            line = int(start.get("line", 0)) + 1

            # reportAttributeAccessIssue is resolution-sensitive on a bare clone (no deps
            # installed). Curate it so it only fires when a symbol is provably absent:
            if rule == "reportAttributeAccessIssue":
                m = _UNKNOWN_IMPORT_RE.search(message)
                if m:
                    # "unknown import symbol" → verify against the real module source.
                    symbol = m.group(1)
                    module = _import_module_for_line(repo_root, rel, line)
                    mod_file = _resolve_module_file(repo_root, module) if module else None
                    if mod_file is None:
                        logger.debug(
                            "[pyright] drop import-symbol FP (module unresolvable/third-party): %s from %s",
                            symbol, module,
                        )
                        continue
                    if _module_binds_symbol(mod_file, symbol):
                        logger.debug(
                            "[pyright] drop import-symbol FP (re-export bound in %s): %s",
                            mod_file, symbol,
                        )
                        continue
                    # genuinely absent from a resolved first-party module → keep
                else:
                    # attribute-on-type subclass is unverifiable from source on a bare
                    # clone → drop from the floor; the LLM + checklist cover real cases.
                    logger.debug("[pyright] drop attribute-on-type finding (lift to LLM): %s", message[:80])
                    continue

            by_path.setdefault(rel, []).append({
                "line": line,
                "message": message,
                "severity": "ERROR",
                "check_id": f"pyright/{rule}",
            })
        return by_path
    finally:
        if ephemeral_config is not None:
            try:
                ephemeral_config.unlink()
            except OSError:
                pass
