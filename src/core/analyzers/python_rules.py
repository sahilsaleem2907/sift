"""Python deterministic verdict analyzers (tree-sitter based).

Currently implements:
- `mutable_default`: a `@dataclass` field whose default is a mutable literal
  (`[]`, `{}`, `set()`, `list()`, `dict()`) not wrapped in `field(default_factory=...)`.
  Python raises `ValueError` at class-creation time for this, so it is a guaranteed
  runtime error — high precision, safe to promote as a critic-exempt floor finding.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from tree_sitter import Parser

from src.intelligence.ast.language_registry import get_language_for_path

from .base import AnalyzerFinding

logger = logging.getLogger(__name__)

_MUTABLE_LITERAL_TYPES = frozenset({"list", "dictionary", "set"})
_MUTABLE_CALL_NAMES = frozenset({"list", "dict", "set"})


def _text(node) -> str:
    t = node.text
    return t.decode("utf-8", errors="replace") if isinstance(t, bytes) else (t or "")


def _line(node) -> int:
    return node.start_point[0] + 1


def _overlaps(line: int, ranges: List[Tuple[int, int]]) -> bool:
    return any(s <= line <= e for s, e in ranges)


def _is_dataclass_decorated(decorated_node) -> bool:
    """True if a `decorated_definition` node carries an @dataclass decorator."""
    for child in decorated_node.children:
        if child.type == "decorator":
            # matches @dataclass and @dataclass(...) and @dataclasses.dataclass
            txt = _text(child).lstrip("@").strip()
            head = txt.split("(", 1)[0].strip()
            if head.split(".")[-1] == "dataclass":
                return True
    return False


def _mutable_default_value(value_node) -> Optional[str]:
    """Return a short description if value_node is a mutable default, else None."""
    if value_node is None:
        return None
    if value_node.type in _MUTABLE_LITERAL_TYPES:
        return _text(value_node)[:40]
    if value_node.type == "call":
        fn = value_node.child_by_field_name("function")
        if fn is not None and fn.type == "identifier" and _text(fn) in _MUTABLE_CALL_NAMES:
            return _text(value_node)[:40]
    return None


def _is_field_default_factory(value_node) -> bool:
    """True for `field(default_factory=...)` — the correct, non-mutable-default form."""
    if value_node is None or value_node.type != "call":
        return False
    fn = value_node.child_by_field_name("function")
    if fn is None:
        return False
    return _text(fn).split(".")[-1] == "field"


def _class_body(class_node):
    return class_node.child_by_field_name("body")


def _scan_dataclass_fields(
    class_node, ranges: List[Tuple[int, int]], out: List[AnalyzerFinding]
) -> None:
    body = _class_body(class_node)
    if body is None:
        return
    for stmt in body.children:
        # Annotated class-level assignment: `name: T = <default>`
        assign = stmt
        if stmt.type == "expression_statement" and stmt.children:
            assign = stmt.children[0]
        if assign.type != "assignment":
            continue
        value_node = assign.child_by_field_name("right")
        if _is_field_default_factory(value_node):
            continue
        desc = _mutable_default_value(value_node)
        if desc is None:
            continue
        line = _line(assign)
        if not _overlaps(line, ranges):
            continue
        left = assign.child_by_field_name("left")
        field_name = _text(left) if left is not None else "field"
        out.append(
            AnalyzerFinding(
                line=line,
                message=(
                    f"Dataclass field '{field_name}' has a mutable default `{desc}`. "
                    f"Python raises ValueError at class creation for mutable dataclass "
                    f"defaults; use `field(default_factory=...)` instead."
                ),
                check_id="analyzer/mutable-default",
            )
        )


def _walk(node, ranges: List[Tuple[int, int]], out: List[AnalyzerFinding]) -> None:
    if node.type == "decorated_definition" and _is_dataclass_decorated(node):
        for child in node.children:
            if child.type == "class_definition":
                _scan_dataclass_fields(child, ranges, out)
    for child in node.children:
        _walk(child, ranges, out)


class PythonAnalyzer:
    lang_key = "python"

    def analyze(
        self, path: str, source: str, changed_ranges: List[Tuple[int, int]]
    ) -> List[AnalyzerFinding]:
        if not source or not changed_ranges:
            return []
        lang = get_language_for_path(path, source)
        if lang is None:
            return []
        parser = Parser()
        parser.set_language(lang)
        tree = parser.parse(source.encode("utf-8"))
        out: List[AnalyzerFinding] = []
        _walk(tree.root_node, changed_ranges, out)
        if out:
            logger.debug(
                "[analyzer] %s: %d python verdict finding(s)", path, len(out)
            )
        return out
