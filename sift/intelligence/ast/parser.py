import logging
from typing import Any, Dict, List, Optional

from tree_sitter import Parser

from .language_registry import detect_language_key, get_language_for_path


logger = logging.getLogger(__name__)


def _node_to_dict(
    node,
    source_bytes: bytes,
    path: str,
    lang_key: str,
    max_text_len: Optional[int],
) -> Dict[str, Any]:
    start_row, start_col = node.start_point
    end_row, end_col = node.end_point

    text: Optional[str] = None
    if max_text_len is not None and max_text_len >= 0:
        span = source_bytes[node.start_byte : node.end_byte]
        if max_text_len and len(span) > max_text_len:
            span = span[:max_text_len]
        text = span.decode("utf-8", errors="replace")

    children: List[Dict[str, Any]] = [
        _node_to_dict(child, source_bytes, path, lang_key, max_text_len)
        for child in node.children
    ]

    return {
        "type": node.type,
        "start_line": start_row + 1,
        "end_line": end_row + 1,
        "start_col": start_col,
        "end_col": end_col,
        "text": text,
        "children": children,
        "lang": lang_key,
        "path": path,
    }


def parse_source(
    path: str,
    source: str,
    max_text_len: Optional[int] = 200,
) -> Optional[Dict[str, Any]]:
    """Parse a file with tree-sitter and return a normalized JSON/dict AST.

    The returned structure is the normalized root node:

    {
        "type": str,
        "start_line": int,
        "end_line": int,
        "start_col": int,
        "end_col": int,
        "text": Optional[str],
        "children": List[Node],
        "lang": str,
        "path": str,
    }

    Returns None if the language cannot be detected or the parser cannot be
    constructed for this file.
    """
    lang = get_language_for_path(path, source)
    if lang is None:
        logger.debug("Skipping AST parse; no language detected for %s", path)
        return None

    lang_key = detect_language_key(path, source) or "unknown"
    logger.debug("Parsing %s with language key %s", path, lang_key)

    parser = Parser()
    parser.set_language(lang)

    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node
    logger.debug(
        "Parsed %s: root type=%s, lines=%s-%s",
        path,
        root.type,
        root.start_point[0] + 1,
        root.end_point[0] + 1,
    )

    return _node_to_dict(root, source_bytes, path, lang_key, max_text_len)


__all__ = ["parse_source"]

