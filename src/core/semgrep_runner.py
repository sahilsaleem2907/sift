"""Run Semgrep on a set of file contents; return findings by path. Used for diff-only context."""
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SEMGREP_TIMEOUT = 120


def _normalize_path(raw_path: str, root: Path) -> str:
    """Convert Semgrep path to repo-relative path string. Handles macOS /var vs /private/var."""
    raw_path = raw_path.replace("\\", "/")
    root_str = str(root)
    root_resolved = str(root.resolve())
    for prefix in (root_resolved + "/", root_resolved, root_str + "/", root_str):
        if raw_path == prefix or raw_path.startswith(prefix):
            suffix = raw_path[len(prefix):].lstrip("/")
            if suffix:
                return suffix.replace("\\", "/")
            break
    if not raw_path.startswith("/"):
        return raw_path
    try:
        p = Path(raw_path)
        if not p.is_absolute():
            return raw_path
        rel = p.relative_to(root)
        return str(rel).replace("\\", "/")
    except (ValueError, TypeError):
        return raw_path


def _parse_result(r: dict, root: Path) -> Optional[Tuple[str, dict]]:
    """Parse a single Semgrep result into (path_str, finding) with full scope. Returns None if no line."""
    raw_path = r.get("path") or ""
    path_str = _normalize_path(raw_path, root)
    start = r.get("start") or {}
    end = r.get("end") or {}
    line = start.get("line")
    if line is None:
        return None
    extra = r.get("extra") or {}
    metadata = extra.get("metadata") or {}
    finding = {
        "line": line,
        "message": extra.get("message") or "",
        "severity": extra.get("severity") or "WARNING",
        "check_id": r.get("check_id") or "",
        "start": start,
        "end": end,
        "extra": {
            "metadata": metadata,
            "message": extra.get("message"),
            "severity": extra.get("severity"),
        },
    }
    return (path_str, finding)


def _parse_error(err: dict, root: Path) -> Optional[Tuple[str, dict]]:
    """Parse a Semgrep error (syntax/parsing) into (path_str, finding). Returns None if no line info."""
    raw_path = err.get("path") or ""
    spans = err.get("spans") or []
    if spans:
        start = spans[0].get("start") or {}
        line = start.get("line")
    else:
        line = None
    if line is None:
        return None
    path_str = _normalize_path(raw_path, root)
    msg = (err.get("message") or "Semgrep parsing/syntax error").strip()
    if raw_path and raw_path in msg:
        msg = msg.replace(raw_path, path_str, 1)
    finding = {
        "line": line,
        "message": msg,
        "severity": "ERROR",
        "check_id": "semgrep-parse-error",
        "start": spans[0].get("start") if spans else {},
        "end": spans[0].get("end") if spans else {},
        "extra": {"metadata": {"type": err.get("type"), "code": err.get("code")}},
    }
    return (path_str, finding)


def run_semgrep(path_to_content: Dict[str, str]) -> Dict[str, List[dict]]:
    """Run Semgrep on the given path->content map. Returns path -> list of findings.

    Each finding has: line, message, severity, check_id, start, end, extra (full scope).
    Syntax/parsing errors from Semgrep's errors array are included as ERROR findings.
    If Semgrep is not available or fails, returns {} so the review can proceed without Semgrep context.
    """
    if not path_to_content:
        return {}

    paths_written = [p for p, c in path_to_content.items() if c]
    logger.debug(
        "Semgrep input: scanning %d file(s): %s",
        len(paths_written),
        paths_written,
    )

    with tempfile.TemporaryDirectory(prefix="sift_semgrep_") as tmpdir:
        root = Path(tmpdir)
        for path, content in path_to_content.items():
            if not content:
                continue
            out_path = root / path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                out_path.write_text(content, encoding="utf-8")
            except (OSError, UnicodeEncodeError) as e:
                logger.debug("Skip writing %s for Semgrep: %s", path, e)
                continue

        try:
            result = subprocess.run(
                ["semgrep", "scan", "--config", "auto", "--json", "--no-git-ignore", str(root)],
                capture_output=True,
                text=True,
                timeout=SEMGREP_TIMEOUT,
                cwd=str(root),
            )
        except FileNotFoundError:
            logger.debug("Semgrep not found in PATH; skipping Semgrep context")
            return {}
        except subprocess.TimeoutExpired:
            logger.warning("Semgrep scan timed out after %ss", SEMGREP_TIMEOUT)
            return {}
        except Exception as e:
            logger.warning("Semgrep scan failed: %s", e)
            return {}

        if result.returncode not in (0, 1):
            logger.warning("Semgrep exited with code %s: %s", result.returncode, result.stderr)
            return {}

        try:
            data = json.loads(result.stdout) if result.stdout else {}
        except json.JSONDecodeError as e:
            logger.warning("Semgrep JSON parse failed: %s", e)
            return {}

        results = data.get("results") or []
        errors = data.get("errors") or []
        by_path: Dict[str, List[dict]] = {}
        for r in results:
            parsed = _parse_result(r, root)
            if parsed:
                path_str, finding = parsed
                by_path.setdefault(path_str, []).append(finding)
        for err in errors:
            parsed = _parse_error(err, root)
            if parsed:
                path_str, finding = parsed
                by_path.setdefault(path_str, []).append(finding)
                logger.debug(
                    "Semgrep error (syntax/parse) added as finding: %s:%s — %s",
                    path_str,
                    finding.get("line"),
                    (finding.get("message") or "")[:80],
                )
        out_summary = {path: len(findings) for path, findings in by_path.items()}
        total = sum(out_summary.values())
        logger.debug(
            "Semgrep output: %d total finding(s) across %d file(s): %s",
            total,
            len(by_path),
            out_summary,
        )
        for path, findings in by_path.items():
            for f in findings:
                msg = (f.get("message") or "")[:80]
                if len(f.get("message") or "") > 80:
                    msg += "..."
                logger.debug(
                    "Semgrep finding: %s:%s [%s] %s — %s",
                    path,
                    f.get("line"),
                    f.get("check_id"),
                    f.get("severity"),
                    msg,
                )
        # Rekey to path_to_content keys (e.g. src/extension.ts) so review_engine lookups succeed
        canonical_keys = list(path_to_content.keys())
        remapped: Dict[str, List[dict]] = {}
        for semgrep_path, findings in by_path.items():
            norm = semgrep_path.replace("\\", "/")
            matched = None
            for key in canonical_keys:
                key_norm = key.replace("\\", "/")
                if norm == key_norm or norm.endswith("/" + key_norm):
                    matched = key
                    break
            if matched:
                remapped.setdefault(matched, []).extend(findings)
            elif norm in canonical_keys:
                remapped.setdefault(norm, []).extend(findings)
        return remapped if remapped else by_path
