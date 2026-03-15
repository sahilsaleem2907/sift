"""Run CodeQL on a repo checkout: create DB, analyze with suite, parse SARIF. Phase 1: full create per run."""
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

# Map file extension / path to CodeQL language (codeql database create --language=)
EXT_TO_CODEQL_LANG: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "javascript",  # TypeScript uses javascript extractor
    ".jsx": "javascript",
    ".tsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".kt": "java",  # Kotlin uses java
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".swift": "swift",
}


def languages_from_paths(paths: List[str]) -> List[str]:
    """Return distinct CodeQL languages for the given file paths."""
    seen: Set[str] = set()
    out: List[str] = []
    for p in paths:
        ext = Path(p).suffix.lower()
        lang = EXT_TO_CODEQL_LANG.get(ext)
        if lang and lang not in seen:
            seen.add(lang)
            out.append(lang)
    return out


def _parse_sarif(sarif_path: Path, source_root: Path) -> Dict[str, List[dict]]:
    """Parse SARIF file; return path -> list of {line, message, severity, check_id} (Semgrep shape)."""
    by_path: Dict[str, List[dict]] = {}
    try:
        data = json.loads(sarif_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("CodeQL SARIF read failed: %s", e)
        return by_path

    runs = data.get("runs") or []
    for run in runs:
        results = run.get("results") or []
        artifacts = run.get("artifacts") or []
        # artifact index -> uri (path)
        uri_by_index: Dict[int, str] = {}
        for i, art in enumerate(artifacts):
            loc = art.get("location") or {}
            uri = (loc.get("uri") or "").replace("file://", "").lstrip("/")
            if uri:
                uri_by_index[i] = uri

        for r in results:
            rule_id = r.get("ruleId") or ""
            msg_obj = r.get("message") or {}
            message = msg_obj.get("text") or ""
            level = (r.get("level") or "warning").lower()
            locations = r.get("locations") or []
            for loc in locations:
                phys = loc.get("physicalLocation") or {}
                art_idx = phys.get("artifactLocation", {}).get("index")
                region = phys.get("region") or {}
                line = region.get("startLine")
                if line is None:
                    continue
                uri = uri_by_index.get(art_idx, "") if art_idx is not None else (phys.get("artifactLocation") or {}).get("uri") or ""
                uri = uri.replace("file://", "").lstrip("/")
                try:
                    full = Path(uri)
                    if full.is_absolute() and source_root.as_posix() in full.as_posix():
                        path_str = str(full.relative_to(source_root)).replace("\\", "/")
                    elif full.is_absolute():
                        path_str = full.name
                    else:
                        path_str = uri.replace("\\", "/")
                except (ValueError, TypeError):
                    path_str = uri
                if not path_str:
                    continue
                by_path.setdefault(path_str, []).append({
                    "line": line,
                    "message": message,
                    "severity": level.upper() if level in ("error", "warning", "note") else "WARNING",
                    "check_id": rule_id,
                })
    return by_path


def run_codeql(
    source_root: Path,
    suite: str,
    languages: List[str],
    timeout: int,
) -> Dict[str, List[dict]]:
    """Run CodeQL: create DB, analyze with suite, return findings by path (Semgrep-like shape).

    source_root: repo checkout (e.g. from repo_cache). Already at PR head.
    suite: default | security-extended | security-and-quality.
    languages: e.g. ['python', 'javascript']. If empty, tries to auto-detect from source_root.
    timeout: total seconds for create + analyze.

    Returns Dict[path, list of {line, message, severity, check_id}]. Returns {} on any failure.
    """
    if not languages:
        # Minimal auto-detect: look for common files
        detected: List[str] = []
        for ext, lang in EXT_TO_CODEQL_LANG.items():
            if next(source_root.rglob(f"*{ext}"), None) and lang not in detected:
                detected.append(lang)
        languages = detected if detected else ["python"]
    languages = list(dict.fromkeys(languages))

    try:
        subprocess.run(
            ["codeql", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.debug("CodeQL not found in PATH; skipping CodeQL")
        return {}

    with tempfile.TemporaryDirectory(prefix="sift_codeql_db_") as db_parent:
        db_path = Path(db_parent) / "db"
        # Single-language: one DB. Multi-language: db-cluster (multiple subdirs).
        if len(languages) == 1:
            lang = languages[0]
            create_cmd = [
                "codeql", "database", "create",
                str(db_path),
                "--language", lang,
                f"--source-root={source_root}",
            ]
        else:
            create_cmd = [
                "codeql", "database", "create",
                str(db_path),
                "--db-cluster",
                f"--source-root={source_root}",
            ]
            for lang in languages:
                create_cmd.extend(["--language", lang])

        try:
            create_result = subprocess.run(
                create_cmd,
                capture_output=True,
                text=True,
                timeout=max(60, timeout - 120),
                cwd=str(source_root),
            )
            logger.debug("CodeQL database create stdout: %s", create_result.stdout)
            logger.debug("CodeQL database create stderr: %s", create_result.stderr)
            if create_result.returncode != 0:
                logger.warning(
                    "CodeQL database create failed: %s %s",
                    create_result.stderr,
                    create_result.stdout,
                )
                return {}
        except subprocess.TimeoutExpired:
            logger.warning("CodeQL database create timed out")
            return {}

        # Analyze: run queries. For db-cluster, we have db/python, db/javascript, etc.
        if len(languages) == 1:
            analyze_dirs = [db_path]
        else:
            analyze_dirs = [db_path / lang for lang in languages if (db_path / lang).exists()]

        all_by_path: Dict[str, List[dict]] = {}
        for adir in analyze_dirs:
            if not adir.exists():
                continue
            out_sarif = Path(db_parent) / f"out_{adir.name}.sarif"
            # Suite: default = no extra arg (use pack default). security-extended / security-and-quality = pass suite.
            analyze_cmd = [
                "codeql", "database", "analyze",
                str(adir),
                "--format=sarif-latest",
                f"--output={out_sarif}",
                "--threads=0",
            ]
            if suite == "security-extended":
                analyze_cmd.append("security-extended")
            elif suite == "security-and-quality":
                analyze_cmd.append("security-and-quality")
            # else default: no query arg uses pack default

            try:
                analyze_result = subprocess.run(
                    analyze_cmd,
                    capture_output=True,
                    text=True,
                    timeout=max(60, timeout - 60),
                )
                logger.debug("CodeQL database analyze stdout: %s", analyze_result.stdout)
                logger.debug("CodeQL database analyze stderr: %s", analyze_result.stderr)
                if analyze_result.returncode != 0:
                    logger.warning(
                        "CodeQL analyze failed for %s: %s",
                        adir.name,
                        analyze_result.stderr or analyze_result.stdout,
                    )
                    continue
            except subprocess.TimeoutExpired:
                logger.warning("CodeQL database analyze timed out for %s", adir.name)
                continue

            if out_sarif.exists():
                for path, findings in _parse_sarif(out_sarif, source_root).items():
                    all_by_path.setdefault(path, []).extend(findings)

    return all_by_path
