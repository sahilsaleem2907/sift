"""Extract function-level chunks from source files for modified diff ranges."""
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tree_sitter import Parser

from .diff_ast import get_new_file_plus_line_ranges
from .language_registry import get_language_for_path

logger = logging.getLogger(__name__)

_FUNCTION_NODE_TYPES = frozenset({
    # Python
    "function_definition",
    # JS / TS / Go / Java / C / C++ / Rust / PHP / Scala / Kotlin / Swift
    "function_declaration",
    "method_definition",
    "method_declaration",
    # Ruby
    "method",
    # Arrow functions (JS/TS) when assigned to a variable are captured via
    # their parent variable_declarator; we also grab standalone arrow_function.
    "arrow_function",
    # Go
    "func_literal",
    # Rust
    "function_item",
    # C# / Kotlin
    "constructor_declaration",
    # Lua
    "function",
})

_WS_COLLAPSE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class FunctionChunk:
    path: str
    name: Optional[str]
    start_line: int
    end_line: int
    text: str
    content_hash: str


def _normalize_text(text: str) -> str:
    return _WS_COLLAPSE_RE.sub(" ", text).strip()


def _content_hash(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _node_name(node) -> Optional[str]:
    """Try to extract the function/method name from a tree-sitter node."""
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return child.text.decode("utf-8", errors="replace") if isinstance(child.text, bytes) else child.text
    if node.type == "arrow_function" and node.parent and node.parent.type == "variable_declarator":
        for child in node.parent.children:
            if child.type in ("identifier", "name"):
                return child.text.decode("utf-8", errors="replace") if isinstance(child.text, bytes) else child.text
    return None


def _overlaps_any_range(
    node_start: int, node_end: int, ranges: List[Tuple[int, int]]
) -> bool:
    for rng_start, rng_end in ranges:
        if node_start <= rng_end and node_end >= rng_start:
            return True
    return False


def _walk_functions(
    node,
    source_bytes: bytes,
    path: str,
    ranges: List[Tuple[int, int]],
    out: List[FunctionChunk],
) -> None:
    """Recursively walk the CST and collect function nodes overlapping ranges."""
    if node.type in _FUNCTION_NODE_TYPES:
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        if _overlaps_any_range(start_line, end_line, ranges):
            text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            name = _node_name(node)
            out.append(FunctionChunk(
                path=path,
                name=name,
                start_line=start_line,
                end_line=end_line,
                text=text,
                content_hash=_content_hash(text),
            ))
            return  # don't recurse into nested functions separately

    for child in node.children:
        _walk_functions(child, source_bytes, path, ranges, out)


def extract_modified_functions(
    path: str, source: str, diff_chunk: str
) -> List[FunctionChunk]:
    """Return FunctionChunks for functions whose spans overlap the diff's '+' lines.

    Uses tree-sitter to parse source and get_new_file_plus_line_ranges to
    determine which lines were added/changed.  Returns an empty list if
    language detection, parsing, or range computation fails.
    """
    if not source or not diff_chunk:
        logger.debug("[Function extract] path=%s: empty source or diff_chunk, skip", path)
        return []

    ranges = get_new_file_plus_line_ranges(diff_chunk)
    if not ranges:
        logger.debug("[Function extract] path=%s: no '+' line ranges in diff, skip", path)
        return []

    logger.debug("[Function extract] path=%s: diff ranges (new-file '+' lines)=%s", path, ranges)
    lang = get_language_for_path(path, source)
    if lang is None:
        logger.debug("[Function extract] path=%s: no language detected, skip", path)
        return []

    parser = Parser()
    parser.set_language(lang)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    chunks: List[FunctionChunk] = []
    _walk_functions(tree.root_node, source_bytes, path, ranges, chunks)

    for c in chunks:
        logger.debug(
            "[Function extract]   chunk path=%s name=%s lines %d-%d hash=%s...",
            c.path, c.name, c.start_line, c.end_line, c.content_hash[:12],
        )
    logger.debug(
        "[Function extract] path=%s: extracted %d function chunk(s) for %d diff range(s)",
        path, len(chunks), len(ranges),
    )
    return chunks


__all__ = ["FunctionChunk", "extract_modified_functions"]
