"""Run Semgrep on a set of file contents; return findings by path. Used for diff-only context."""
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

SEMGREP_TIMEOUT = 120


def run_semgrep(path_to_content: Dict[str, str]) -> Dict[str, List[dict]]:
    """Run Semgrep on the given path->content map. Returns path -> list of findings.

    Each finding is a dict with keys: line, message, severity, check_id.
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
        by_path: Dict[str, List[dict]] = {}
        for r in results:
            raw_path = r.get("path") or ""
            try:
                p = Path(raw_path)
                rel = p.relative_to(root)
                path_str = str(rel).replace("\\", "/")
            except (ValueError, TypeError):
                path_str = raw_path.replace("\\", "/")
            start = r.get("start") or {}
            line = start.get("line")
            if line is None:
                continue
            extra = r.get("extra") or {}
            by_path.setdefault(path_str, []).append({
                "line": line,
                "message": extra.get("message") or "",
                "severity": extra.get("severity") or "WARNING",
                "check_id": r.get("check_id") or "",
            })
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
        return by_path
