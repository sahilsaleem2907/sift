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
    name = Path(path).name.lower()
    # Dockerfile: basename or .dockerfile suffix
    if name == "dockerfile" or p.endswith(".dockerfile"):
        return "hadolint"
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
    if p.endswith(".rb"):
        return "rubocop"
    if p.endswith(".rs"):
        return "rustc"
    if p.endswith((".c", ".h")):
        return "cppcheck"
    if p.endswith((".cpp", ".cc", ".cxx", ".hpp")):
        return "cppcheck"
    if p.endswith(".cs"):
        return "csharp"
    if p.endswith(".php"):
        return "phpstan"
    if p.endswith(".swift"):
        return "swiftlint"
    if p.endswith((".kt", ".kts")):
        return "ktlint"
    if p.endswith((".sh", ".bash", ".zsh")):
        return "shellcheck"
    if p.endswith(".css"):
        return "stylelint"
    if p.endswith((".scss", ".sass")):
        return "stylelint"
    if p.endswith((".yml", ".yaml")):
        return "yamllint"
    if p.endswith((".tf", ".tfvars")):
        return "tflint"
    if p.endswith(".lua"):
        return "luacheck"
    if p.endswith((".ex", ".exs")):
        return "elixirc"
    if p.endswith(".r"):
        return "lintr"
    if p.endswith((".pl", ".pm")):
        return "perlcritic"
    if p.endswith((".md", ".mdx")):
        return "markdownlint"
    if p.endswith(".json"):
        return "json_syntax"
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


# mcs (C#): file(line,col): error CS1234: message
_MCS_LINE_RE = re.compile(
    r"^(.+?)\((\d+),(\d+)\):\s*(error|warning)\s+(CS\d+)?:\s*(.+)$", re.MULTILINE
)
# yamllint -f parsable: path:line:col: [level] message
_YAMLLINT_RE = re.compile(
    r"^([^:]+):(\d+):(\d+):\s*\[(error|warning|info)\]\s*(.+)$", re.MULTILINE
)
# luacheck plain: file:line:col: (W/E) message
_LUACHECK_RE = re.compile(
    r"^([^:]+):(\d+):(\d+):\s*\(([WE])\d+\)\s*(.+)$", re.MULTILINE
)
# elixirc: file:line: (error|warning) message
_ELIXIRC_RE = re.compile(
    r"^([^:]+):(\d+):\s*(error|warning):\s*(.+)$", re.MULTILINE
)
# R lintr: file:line:col: [type] message
_LINTR_RE = re.compile(
    r"^([^:]+):(\d+):(\d+):\s*([^:]+):\s*(.+)$", re.MULTILINE
)
# perlcritic verbose "%l %c %p %m": line column policy message
_PERLCRITIC_RE = re.compile(
    r"^(\d+)\s+(\d+)\s+([^\s]+)\s+(.+)$", re.MULTILINE
)
# json.tool stderr: "..." line N or line N column M
_JSON_TOOL_LINE_RE = re.compile(r"line\s+(\d+)", re.IGNORECASE)


def _run_rubocop(root: Path, path: str) -> List[LinterIssue]:
    """Run rubocop on a single file; return list of unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["rubocop", "--format", "json", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("Rubocop not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("Rubocop timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("Rubocop failed for %s: %s", path, e)
        return out
    raw = result.stdout.strip()
    if not raw:
        return out
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return out
    if not isinstance(data, dict):
        return out
    path_norm = path.replace("\\", "/")
    for file_data in data.get("files") or []:
        if not isinstance(file_data, dict):
            continue
        if (file_data.get("path") or "").replace("\\", "/") != path_norm and path_norm not in (file_data.get("path") or ""):
            continue
        for off in file_data.get("offenses") or []:
            if not isinstance(off, dict):
                continue
            loc = off.get("location") or {}
            line = loc.get("start_line") or loc.get("line")
            if line is None:
                continue
            msg = (off.get("message") or "").strip()
            sev = (off.get("severity") or "warning").lower()
            severity = "error" if sev == "error" else "warning" if sev == "warning" else "info"
            out.append({
                "line": line,
                "message": msg,
                "severity": severity,
                "rule_id": off.get("cop_name") or "",
                "source": "rubocop",
            })
    return out


def _run_rustc(root: Path, path: str) -> List[LinterIssue]:
    """Run rustc --emit=metadata on a single file; parse JSON stderr for diagnostics."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["rustc", "--edition=2021", "--error-format=json", "--emit=metadata", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("rustc not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("rustc timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("rustc failed for %s: %s", path, e)
        return out
    base_name = Path(path).name
    for line_str in (result.stderr or "").strip().splitlines():
        line_str = line_str.strip()
        if not line_str:
            continue
        try:
            obj = json.loads(line_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        msg_obj = obj.get("message")
        if not isinstance(msg_obj, dict):
            continue
        level = (msg_obj.get("level") or "error").lower()
        severity = "error" if level == "error" else "warning" if level == "warning" else "info"
        message = (msg_obj.get("message") or "").strip()
        code = msg_obj.get("code")
        rule_id = (code.get("code") or "") if isinstance(code, dict) else ""
        for span in msg_obj.get("spans") or []:
            if not isinstance(span, dict):
                continue
            line = span.get("line_start") or span.get("line")
            if line is None:
                continue
            span_path = span.get("file_name") or ""
            if base_name not in span_path and path not in span_path:
                continue
            out.append({
                "line": line,
                "message": message,
                "severity": severity,
                "rule_id": rule_id,
                "source": "rustc",
            })
    return out


def _run_cppcheck(root: Path, path: str) -> List[LinterIssue]:
    """Run cppcheck on a single file; parse XML output for unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["cppcheck", "--xml", "--xml-version=2", "--enable=warning,style", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("cppcheck not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("cppcheck timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("cppcheck failed for %s: %s", path, e)
        return out
    # cppcheck writes XML to stderr
    text = (result.stderr or "").strip()
    if not text:
        return out
    try:
        root_el = ET.fromstring(text)
    except ET.ParseError:
        return out
    path_norm = path.replace("\\", "/")
    for err in root_el.findall(".//error"):
        loc = err.find("location")
        if loc is None:
            continue
        file_attr = (loc.get("file") or "").replace("\\", "/")
        if path_norm not in file_attr and Path(path).name not in file_attr:
            continue
        try:
            line = int(loc.get("line", 0))
        except (TypeError, ValueError):
            continue
        if line <= 0:
            continue
        msg = (err.get("msg") or "").strip()
        sev = (err.get("severity") or "warning").lower()
        severity = "error" if sev == "error" else "warning" if sev == "warning" else "info"
        out.append({
            "line": line,
            "message": msg,
            "severity": severity,
            "rule_id": err.get("id") or "",
            "source": "cppcheck",
        })
    return out


def _run_csharp(root: Path, path: str) -> List[LinterIssue]:
    """Run mcs (Mono C# compiler) on a single file; parse stderr for line/message."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["mcs", "-nologo", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("mcs not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("mcs timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("mcs failed for %s: %s", path, e)
        return out
    text = (result.stderr or "").strip()
    base_name = Path(path).name
    for m in _MCS_LINE_RE.finditer(text):
        file_part, line_str, _col, level, code, message = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)
        if base_name not in file_part and path not in file_part:
            continue
        try:
            line = int(line_str)
        except ValueError:
            continue
        severity = "error" if (level or "").lower() == "error" else "warning"
        out.append({
            "line": line,
            "message": (message or "").strip(),
            "severity": severity,
            "rule_id": code or "",
            "source": "mcs",
        })
    return out


def _run_phpstan(root: Path, path: str) -> List[LinterIssue]:
    """Run phpstan analyse on a single file; return list of unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["phpstan", "analyse", "--error-format=json", "--no-progress", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("phpstan not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("phpstan timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("phpstan failed for %s: %s", path, e)
        return out
    raw = result.stdout.strip()
    if not raw:
        return out
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return out
    if not isinstance(data, dict):
        return out
    for msg in data.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        line = msg.get("line")
        if line is None:
            continue
        message = (msg.get("message") or "").strip()
        out.append({
            "line": line,
            "message": message,
            "severity": "error",
            "rule_id": msg.get("identifier") or "",
            "source": "phpstan",
        })
    return out


def _run_swiftlint(root: Path, path: str) -> List[LinterIssue]:
    """Run swiftlint lint on a single file; return list of unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["swiftlint", "lint", "--reporter", "json", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("swiftlint not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("swiftlint timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("swiftlint failed for %s: %s", path, e)
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
        out.append({
            "line": line,
            "message": (item.get("reason") or item.get("message") or "").strip(),
            "severity": "warning",
            "rule_id": item.get("rule_id") or "",
            "source": "swiftlint",
        })
    return out


def _run_ktlint(root: Path, path: str) -> List[LinterIssue]:
    """Run ktlint on a single file; return list of unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["ktlint", "--reporter=json", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("ktlint not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("ktlint timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("ktlint failed for %s: %s", path, e)
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
    path_norm = path.replace("\\", "/")
    for item in data:
        if not isinstance(item, dict):
            continue
        if (item.get("file") or "").replace("\\", "/") != path_norm and path_norm not in (item.get("file") or ""):
            continue
        line = item.get("line") or item.get("startLine")
        if line is None:
            continue
        out.append({
            "line": line,
            "message": (item.get("message") or item.get("description") or "").strip(),
            "severity": "warning",
            "rule_id": item.get("ruleId") or item.get("rule") or "",
            "source": "ktlint",
        })
    return out


def _run_shellcheck(root: Path, path: str) -> List[LinterIssue]:
    """Run shellcheck on a single file; return list of unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["shellcheck", "-f", "json", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("shellcheck not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("shellcheck timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("shellcheck failed for %s: %s", path, e)
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
        level = (item.get("level") or "warning").lower()
        severity = "error" if level == "error" else "warning" if level == "warning" else "info"
        out.append({
            "line": line,
            "message": (item.get("message") or "").strip(),
            "severity": severity,
            "rule_id": item.get("code") or "",
            "source": "shellcheck",
        })
    return out


def _run_stylelint(root: Path, path: str) -> List[LinterIssue]:
    """Run stylelint on a single file (CSS/SCSS); return list of unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["stylelint", "--formatter", "json", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("stylelint not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("stylelint timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("stylelint failed for %s: %s", path, e)
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
        if (file_result.get("source") or "").replace("\\", "/") != path.replace("\\", "/"):
            continue
        for msg in file_result.get("warnings") or file_result.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            line = msg.get("line")
            if line is None:
                continue
            sev = msg.get("severity") or "warning"
            severity = "error" if sev == "error" else "warning"
            out.append({
                "line": line,
                "message": (msg.get("text") or msg.get("message") or "").strip(),
                "severity": severity,
                "rule_id": msg.get("rule") or msg.get("ruleId") or "",
                "source": "stylelint",
            })
    return out


def _run_yamllint(root: Path, path: str) -> List[LinterIssue]:
    """Run yamllint with parsable format; parse text output into unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["yamllint", "-f", "parsable", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("yamllint not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("yamllint timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("yamllint failed for %s: %s", path, e)
        return out
    text = (result.stdout or result.stderr or "").strip()
    path_norm = path.replace("\\", "/")
    base_name = Path(path).name
    for m in _YAMLLINT_RE.finditer(text):
        file_part, line_str, _col, level, message = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        file_part = file_part.replace("\\", "/")
        if not file_part.endswith(path_norm) and base_name not in file_part and path_norm not in file_part:
            continue
        try:
            line = int(line_str)
        except ValueError:
            continue
        severity = "error" if level == "error" else "warning" if level == "warning" else "info"
        # Optional (rule_id) at end of message
        rule_id = ""
        if " (" in message and message.endswith(")"):
            idx = message.rfind(" (")
            rule_id = message[idx + 2 : -1].strip()
            message = message[:idx].strip()
        out.append({
            "line": line,
            "message": message,
            "severity": severity,
            "rule_id": rule_id,
            "source": "yamllint",
        })
    return out


def _run_hadolint(root: Path, path: str) -> List[LinterIssue]:
    """Run hadolint on a Dockerfile; return list of unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["hadolint", "--format", "json", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("hadolint not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("hadolint timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("hadolint failed for %s: %s", path, e)
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
        out.append({
            "line": line,
            "message": (item.get("message") or "").strip(),
            "severity": "warning",
            "rule_id": item.get("code") or item.get("rule") or "",
            "source": "hadolint",
        })
    return out


def _run_tflint(root: Path, path: str) -> List[LinterIssue]:
    """Run tflint on a Terraform file; return list of unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["tflint", "--format", "json", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("tflint not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("tflint timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("tflint failed for %s: %s", path, e)
        return out
    raw = result.stdout.strip()
    if not raw:
        return out
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return out
    issues_list = data.get("issues", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    for item in issues_list:
        if not isinstance(item, dict):
            continue
        range_obj = item.get("range") or item.get("Range") or {}
        start = range_obj.get("start") or range_obj.get("Start") or {}
        line = start.get("line") or start.get("Line")
        if line is None:
            line = item.get("line")
        if line is None:
            continue
        rule = item.get("rule") or item.get("Rule") or {}
        rule_name = rule.get("name", "") or rule.get("Name", "") if isinstance(rule, dict) else ""
        out.append({
            "line": line,
            "message": (item.get("message") or "").strip(),
            "severity": "warning",
            "rule_id": rule_name,
            "source": "tflint",
        })
    return out


def _run_luacheck(root: Path, path: str) -> List[LinterIssue]:
    """Run luacheck with plain formatter; parse text into unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["luacheck", "--formatter", "plain", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("luacheck not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("luacheck timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("luacheck failed for %s: %s", path, e)
        return out
    text = (result.stdout or result.stderr or "").strip()
    path_norm = path.replace("\\", "/")
    base_name = Path(path).name
    for m in _LUACHECK_RE.finditer(text):
        file_part, line_str, _col, kind, message = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        file_part = file_part.replace("\\", "/")
        if not file_part.endswith(path_norm) and base_name not in file_part and path_norm not in file_part:
            continue
        try:
            line = int(line_str)
        except ValueError:
            continue
        severity = "error" if kind == "E" else "warning"
        out.append({
            "line": line,
            "message": (message or "").strip(),
            "severity": severity,
            "rule_id": "",
            "source": "luacheck",
        })
    return out


def _run_elixirc(root: Path, path: str) -> List[LinterIssue]:
    """Run elixirc (compile only) on a single file; parse stderr for line/message."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["elixirc", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("elixirc not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("elixirc timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("elixirc failed for %s: %s", path, e)
        return out
    text = (result.stderr or "").strip()
    path_norm = path.replace("\\", "/")
    base_name = Path(path).name
    for m in _ELIXIRC_RE.finditer(text):
        file_part, line_str, level, message = m.group(1), m.group(2), m.group(3), m.group(4)
        file_part = file_part.replace("\\", "/")
        if not file_part.endswith(path_norm) and base_name not in file_part and path_norm not in file_part:
            continue
        try:
            line = int(line_str)
        except ValueError:
            continue
        severity = "error" if level == "error" else "warning"
        out.append({
            "line": line,
            "message": (message or "").strip(),
            "severity": severity,
            "rule_id": "",
            "source": "elixirc",
        })
    return out


def _run_lintr(root: Path, path: str) -> List[LinterIssue]:
    """Run R lintr::lint on a single file; parse text output into unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["Rscript", "-e", f"lintr::lint('{full}')"],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("Rscript/lintr not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("lintr timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("lintr failed for %s: %s", path, e)
        return out
    text = (result.stdout or result.stderr or "").strip()
    path_norm = path.replace("\\", "/")
    base_name = Path(path).name
    for m in _LINTR_RE.finditer(text):
        file_part, line_str, _col, rule_type, message = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        file_part = file_part.replace("\\", "/")
        if not file_part.endswith(path_norm) and base_name not in file_part and path_norm not in file_part:
            continue
        try:
            line = int(line_str)
        except ValueError:
            continue
        out.append({
            "line": line,
            "message": (message or "").strip(),
            "severity": "warning",
            "rule_id": (rule_type or "").strip(),
            "source": "lintr",
        })
    return out


def _run_perlcritic(root: Path, path: str) -> List[LinterIssue]:
    """Run perlcritic on a single file; parse verbose output into unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["perlcritic", "--severity", "1", "--verbose", "%l %c %p %m\n", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("perlcritic not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("perlcritic timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("perlcritic failed for %s: %s", path, e)
        return out
    text = (result.stdout or result.stderr or "").strip()
    for line_str in text.splitlines():
        m = _PERLCRITIC_RE.match(line_str.strip())
        if not m:
            continue
        try:
            line = int(m.group(1))
        except ValueError:
            continue
        out.append({
            "line": line,
            "message": (m.group(4) or "").strip(),
            "severity": "warning",
            "rule_id": (m.group(3) or "").strip(),
            "source": "perlcritic",
        })
    return out


def _run_markdownlint(root: Path, path: str) -> List[LinterIssue]:
    """Run markdownlint on a single file; return list of unified issues."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["markdownlint", "--dot", "--json", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("markdownlint not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("markdownlint timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("markdownlint failed for %s: %s", path, e)
        return out
    raw = (result.stdout or result.stderr or "").strip()
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
        line = item.get("lineNumber") or item.get("line") or item.get("lineNumber")
        if line is None:
            continue
        rule_id = item.get("ruleNames") or item.get("rule") or item.get("ruleName") or ""
        if isinstance(rule_id, list):
            rule_id = ",".join(str(x) for x in rule_id)
        out.append({
            "line": line,
            "message": (item.get("ruleDescription") or item.get("message") or item.get("description") or "").strip(),
            "severity": "warning",
            "rule_id": str(rule_id),
            "source": "markdownlint",
        })
    return out


def _run_json_syntax(root: Path, path: str) -> List[LinterIssue]:
    """Run python3 -m json.tool for syntax-only check; parse stderr for line number."""
    out: List[LinterIssue] = []
    full = root / path
    if not full.exists():
        return out
    try:
        result = subprocess.run(
            ["python3", "-m", "json.tool", str(full)],
            capture_output=True,
            text=True,
            timeout=LINTER_TIMEOUT_PER_FILE,
            cwd=str(root),
        )
    except FileNotFoundError:
        logger.debug("python3 not found in PATH for %s", path)
        return out
    except subprocess.TimeoutExpired:
        logger.debug("json.tool timed out for %s", path)
        return out
    except Exception as e:
        logger.debug("json.tool failed for %s: %s", path, e)
        return out
    if result.returncode == 0:
        return out
    text = (result.stderr or "").strip()
    m = _JSON_TOOL_LINE_RE.search(text)
    line = 1
    if m:
        try:
            line = int(m.group(1))
        except ValueError:
            pass
    out.append({
        "line": line,
        "message": text or "Invalid JSON",
        "severity": "error",
        "rule_id": "",
        "source": "json.tool",
    })
    return out


def run_linters(path_to_content: Dict[str, str]) -> Dict[str, List[LinterIssue]]:
    """Run language-appropriate linters on each path. Return path -> list of unified issues.

    Each issue has: line, message, severity (optional), rule_id (optional), source.
    Supported sources: pylint, eslint, tsc, go vet, spotbugs, rubocop, rustc, cppcheck,
    mcs, phpstan, swiftlint, ktlint, shellcheck, stylelint, yamllint, hadolint, tflint,
    luacheck, elixirc, lintr, perlcritic, markdownlint, json.tool.
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
            elif linter == "rubocop":
                issues = _run_rubocop(root, path)
            elif linter == "rustc":
                issues = _run_rustc(root, path)
            elif linter == "cppcheck":
                issues = _run_cppcheck(root, path)
            elif linter == "csharp":
                issues = _run_csharp(root, path)
            elif linter == "phpstan":
                issues = _run_phpstan(root, path)
            elif linter == "swiftlint":
                issues = _run_swiftlint(root, path)
            elif linter == "ktlint":
                issues = _run_ktlint(root, path)
            elif linter == "shellcheck":
                issues = _run_shellcheck(root, path)
            elif linter == "stylelint":
                issues = _run_stylelint(root, path)
            elif linter == "yamllint":
                issues = _run_yamllint(root, path)
            elif linter == "hadolint":
                issues = _run_hadolint(root, path)
            elif linter == "tflint":
                issues = _run_tflint(root, path)
            elif linter == "luacheck":
                issues = _run_luacheck(root, path)
            elif linter == "elixirc":
                issues = _run_elixirc(root, path)
            elif linter == "lintr":
                issues = _run_lintr(root, path)
            elif linter == "perlcritic":
                issues = _run_perlcritic(root, path)
            elif linter == "markdownlint":
                issues = _run_markdownlint(root, path)
            elif linter == "json_syntax":
                issues = _run_json_syntax(root, path)

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
