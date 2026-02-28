"""Run language-specific linters on file contents; return unified issues by path. Used for diff-only context."""
import json
import logging
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LINTER_TIMEOUT_PER_FILE = 45

# Unified issue shape: line, message, severity?, rule_id?, source
LinterIssue = Dict[str, Any]


def _detect_linter(path: str) -> Optional[str]:
    """Return linter key for path, or None if unsupported."""
    p = path.lower()
    if p.endswith(".py"):
        return "pylint"
    if p.endswith((".js", ".mjs", ".cjs")):
        return "eslint"
    if p.endswith((".ts", ".tsx")):
        return "ts"
    if p.endswith(".go"):
        return "go"
    if p.endswith(".java"):
        return "java"
    return None


def _run_pylint(root: Path, path: str) -> List[LinterIssue]:
    """Run pylint on a single file; return list of unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["pylint", "--output-format=json", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("Pylint not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("Pylint timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("Pylint failed for %s: %s", path, e)
        return out

    raw = result.stdout.strip()
    if not raw:
        return out
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return out
    if not isinstance(data, list):
        return out
    for item in data:
        if not isinstance(item, dict):
            continue
        line = item.get("line")
        if line is None:
            continue
        msg = (item.get("msg") or "").strip()
        symbol = item.get("symbol") or item.get("msg_id") or ""
        # type: convention=1, refactor=2, warning=3, error=4
        severity_map = {"convention": "info", "refactor": "info", "warning": "warning", "error": "error"}
        severity = severity_map.get(item.get("type", ""), "warning")
        out.append({
            "line": line,
            "message": msg,
            "severity": severity,
            "rule_id": symbol,
            "source": "pylint",
        })
    return out


def _run_eslint(root: Path, path: str) -> List[LinterIssue]:
    """Run eslint on a single file; return list of unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["eslint", "--format", "json", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("ESLint not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("ESLint timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("ESLint failed for %s: %s", path, e)
        return out

    raw = result.stdout.strip()
    if not raw:
        return out
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return out
    if not isinstance(data, list):
        return out
    for file_result in data:
        if not isinstance(file_result, dict):
            continue
        for msg_obj in file_result.get("messages") or []:
            if not isinstance(msg_obj, dict):
                continue
            line = msg_obj.get("line")
            if line is None:
                continue
            message = (msg_obj.get("message") or "").strip()
            rule_id = msg_obj.get("ruleId") or ""
            sev = msg_obj.get("severity")
            severity = "error" if sev == 2 else "warning"
            out.append({
                "line": line,
                "message": message,
                "severity": severity,
                "rule_id": rule_id,
                "source": "eslint",
            })
    return out


# tsc stderr: path(line,col): error TS1234: message
_TSC_LINE_RE = re.compile(r"^(.+?)\((\d+),(\d+)\):\s*(?:error|warning)\s+(TS\d+)?:\s*(.+)$", re.MULTILINE)


def _run_tsc(root: Path, path: str) -> List[LinterIssue]:
    """Run tsc --noEmit on a single file; parse stderr for line/col/message."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["tsc", "--noEmit", "--pretty", "false", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("tsc not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("tsc timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("tsc failed for %s: %s", path, e)
        return out

    text = (result.stderr or "").strip()
    base_name = Path(path).name
    for m in _TSC_LINE_RE.finditer(text):
        file_part, line_str, _col, code, message = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        if base_name not in file_part and path not in file_part:
            continue
        try:
            line = int(line_str)
        except ValueError:
            continue
        out.append({
            "line": line,
            "message": (message or "").strip(),
            "severity": "error",
            "rule_id": code or "",
            "source": "tsc",
        })
    return out


# go vet: file:line: message (relative to cwd)
_GOVET_LINE_RE = re.compile(r"^([^:]+):(\d+):\s*(.+)$", re.MULTILINE)


def _run_go_vet(root: Path, path: str) -> List[LinterIssue]:
    """Run go vet in root; parse stderr for file:line: message; return issues for this path only."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    # Ensure go.mod so go vet works
    go_mod = root / "go.mod"
    if not go_mod.exists():
        try:
            go_mod.write_text("module m\n", encoding="utf-8")
        except OSError as e:
            logger.debug("Could not write go.mod for go vet: %s", e)
            return out
    try:
        result = subprocess.run(
            ["go", "vet", "./..."],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("go not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("go vet timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("go vet failed for %s: %s", path, e)
        return out

    text = (result.stderr or "").strip()
    path_norm = path.replace("\\", "/")
    for m in _GOVET_LINE_RE.finditer(text):
        file_part, line_str, message = m.group(1), m.group(2), m.group(3)
        file_part = file_part.replace("\\", "/")
        if not file_part.endswith(path_norm) and path_norm not in file_part:
            continue
        try:
            line = int(line_str)
        except ValueError:
            continue
        out.append({
            "line": line,
            "message": (message or "").strip(),
            "severity": "warning",
            "rule_id": "",
            "source": "go vet",
        })
    return out


def _run_spotbugs(root: Path, path: str, path_to_content: Dict[str, str]) -> List[LinterIssue]:
    """Compile Java file in temp dir, run SpotBugs on classes; return issues for this path only."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    class_dir = root / "out"
    class_dir.mkdir(parents=True, exist_ok=True)
    try:
        comp = subprocess.run(
            ["javac", "-d", str(class_dir), str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("javac not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("javac timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("javac failed for %s: %s", path, e)
        return out

    if comp.returncode != 0:
        # Parse javac stderr for line numbers as fallback
        for line in (comp.stderr or "").splitlines():
            # pattern: path:line: message
            m = re.match(r"^(.+):(\d+):\s*(.+)$", line)
            if m:
                try:
                    line_no = int(m.group(2))
                    out.append({
                        "line": line_no,
                        "message": (m.group(3) or "").strip(),
                        "severity": "error",
                        "rule_id": "javac",
                        "source": "spotbugs",
                    })
                except ValueError:
                    pass
        return out

    try:
        sb = subprocess.run(
            ["spotbugs", "-textui", "-xml", "-output", str(root / "spotbugs.xml"), str(class_dir)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("SpotBugs not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("SpotBugs timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("SpotBugs failed for %s: %s", path, e)
        return out

    xml_path = root / "spotbugs.xml"
    if not xml_path.exists():
        return out
    try:
        tree = ET.parse(xml_path)
        xml_root = tree.getroot()
    except ET.ParseError:
        return out
    path_basename = Path(path).name
    for bug in xml_root.findall(".//BugInstance"):
        src = bug.find("SourceLine")
        if src is None:
            continue
        start = src.get("start")
        source_path = src.get("sourcepath") or ""
        if path_basename not in source_path and path not in source_path:
            continue
        try:
            line = int(start) if start else None
        except (TypeError, ValueError):
            line = None
        if line is None:
            continue
        msg_el = bug.find("LongMessage")
        message = (msg_el.text or "").strip() if msg_el is not None else ""
        if not message and bug.find("ShortMessage") is not None:
            message = (bug.find("ShortMessage").text or "").strip()
        out.append({
            "line": line,
            "message": message,
            "severity": "warning",
            "rule_id": bug.get("type", ""),
            "source": "spotbugs",
        })
    return out


def run_linters(path_to_content: Dict[str, str]) -> Dict[str, List[LinterIssue]]:
    """Run language-appropriate linters on each path. Return path -> list of unified issues.

    Each issue has: line, message, severity (optional), rule_id (optional), source (pylint|eslint|tsc|go vet|spotbugs).
    If a linter is missing or fails, that file gets no issues (log and continue).
    """
    if not path_to_content:
        return {}

    paths_with_content = [p for p, c in path_to_content.items() if c]
    logger.debug(
        "Linter run starting: %d file(s) to process: %s",
        len(paths_with_content),
        paths_with_content,
    )
    by_path: Dict[str, List[LinterIssue]] = {}
    with tempfile.TemporaryDirectory(prefix="sift_linter_") as tmpdir:
        root = Path(tmpdir)
        for path, content in path_to_content.items():
            if not content:
                continue
            linter = _detect_linter(path)
            if not linter:
                logger.debug("Linter skip (unsupported extension): %s", path)
                continue
            logger.debug("Linter running %s on %s", linter, path)
            out_path = root / path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                out_path.write_text(content, encoding="utf-8")
            except (OSError, UnicodeEncodeError) as e:
                logger.debug("Skip writing %s for linter: %s", path, e)
                continue

            issues: List[LinterIssue] = []
            if linter == "pylint":
                issues = _run_pylint(root, path)
            elif linter == "eslint":
                issues = _run_eslint(root, path)
            elif linter == "ts":
                issues = _run_eslint(root, path)
                issues.extend(_run_tsc(root, path))
            elif linter == "go":
                issues = _run_go_vet(root, path)
            elif linter == "java":
                issues = _run_spotbugs(root, path, path_to_content)

            if issues:
                by_path[path] = issues
                logger.debug(
                    "Linter %s: %s produced %d issue(s): %s",
                    linter,
                    path,
                    len(issues),
                    [(i.get("line"), (i.get("rule_id") or i.get("message", ""))[:40]) for i in issues[:5]],
                )
            else:
                logger.debug("Linter %s: %s produced 0 issues (ok or no findings)", linter, path)

    logger.debug(
        "Linter run finished: %d path(s) with issues, %d total",
        len(by_path),
        sum(len(v) for v in by_path.values()),
    )
    return by_path
