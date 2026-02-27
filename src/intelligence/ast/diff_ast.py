import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .parser import parse_source

logger = logging.getLogger(__name__)

_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def _compute_new_line_ranges(diff_chunk: str) -> List[Tuple[int, int]]:
    """Return merged (start, end) ranges for *added/changed* new-file lines.

    Unlike the simpler header-based approach, this walks each hunk and tracks the
    new-file line counter so we only consider lines that are actually added/changed
    (\"+\" lines in the unified diff), not the entire hunk context.
    """
    if not diff_chunk or not diff_chunk.strip():
        return []

    new_lines: List[int] = []
    current_new_line: Optional[int] = None

    for line in diff_chunk.splitlines():
        m = _DIFF_HUNK_RE.match(line)
        if m:
            # Start of a new hunk; reset the new-file line counter.
            current_new_line = int(m.group(1))
            continue

        if current_new_line is None:
            # Not inside a hunk yet.
            continue

        if not line:
            continue

        prefix = line[0]
        if prefix == " ":
            # Context line: advances both old and new counters;
            # we don't record it as a changed line.
            current_new_line += 1
        elif prefix == "+" and not line.startswith("+++"):
            # Added/changed line in the new file.
            new_lines.append(current_new_line)
            current_new_line += 1
        elif prefix == "-" and not line.startswith("---"):
            # Removed line: advances old counter only; new counter unchanged.
            # We don't need to track the old counter explicitly here.
            continue
        else:
            # Headers (--- / +++) or other diff metadata within hunks; ignore.
            continue

    if not new_lines:
        return []

    # Merge contiguous new lines into ranges.
    new_lines_sorted = sorted(set(new_lines))
    ranges: List[Tuple[int, int]] = []
    start = end = new_lines_sorted[0]
    for ln in new_lines_sorted[1:]:
        if ln == end + 1:
            end = ln
        else:
            ranges.append((start, end))
            start = end = ln
    ranges.append((start, end))
    return ranges


def _node_overlaps_ranges(node: Dict[str, Any], ranges: List[Tuple[int, int]]) -> bool:
    ns = node.get("start_line")
    ne = node.get("end_line")
    if ns is None or ne is None:
        return False
    # We only keep nodes that are fully contained within a changed range,
    # so we never send nodes that extend outside the diff hunks (e.g. the
    # top-level program node covering the whole file).
    for start, end in ranges:
        if ns >= start and ne <= end:
            return True
    return False


def _collect_overlapping_nodes(
    node: Dict[str, Any],
    ranges: List[Tuple[int, int]],
    out: List[Dict[str, Any]],
) -> None:
    """Collect nodes that overlap diff ranges. Appends shallow copies without children
    so we only send metadata for lines in the diff, not the full file tree."""
    if _node_overlaps_ranges(node, ranges):
        shallow = {k: v for k, v in node.items() if k != "children"}
        shallow["children"] = []
        out.append(shallow)
    for child in node.get("children") or []:
        _collect_overlapping_nodes(child, ranges, out)


def build_diff_ast(path: str, source: str, diff_chunk: str) -> Optional[Dict[str, Any]]:
    """Build an AST view focused on the diff hunks for a file.

    Returns:
        {
            "path": str,
            "lang": str,
            "changed_ranges": [{"start_line": int, "end_line": int}, ...],
            "nodes": [<normalized AST nodes overlapping changed_ranges>],
        }
    or None if parsing or language detection fails.
    """
    ranges = _compute_new_line_ranges(diff_chunk)
    if not ranges:
        logger.debug("No diff hunk ranges for %s; skipping diff AST", path)
        return None

    ast_root = parse_source(path, source)
    if ast_root is None:
        logger.debug("No AST root for %s; skipping diff AST", path)
        return None

    overlapping: List[Dict[str, Any]] = []
    _collect_overlapping_nodes(ast_root, ranges, overlapping)
    if not overlapping:
        logger.debug("No AST nodes overlap diff ranges for %s", path)
        return None

    lang = ast_root.get("lang") or "unknown"
    changed_ranges = [
        {"start_line": start, "end_line": end} for start, end in ranges
    ]
    return {
        "path": path,
        "lang": lang,
        "changed_ranges": changed_ranges,
        "nodes": overlapping,
    }


__all__ = ["build_diff_ast"]

