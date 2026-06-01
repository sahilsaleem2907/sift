"""PR-internal import graph: detect when changed files are imported by other files in the same PR."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tree_sitter import Parser

from src.intelligence.ast.function_extract import FunctionChunk
from src.intelligence.ast.language_registry import get_language_for_path

logger = logging.getLogger(__name__)

# tree-sitter node types that carry import module paths
_IMPORT_NODE_TYPES = frozenset({
    "import_statement",
    "import_from_statement",
    "import_declaration",
})

_PYTHON_MODULE_CHILD_TYPES = frozenset({
    "dotted_name",
    "relative_import",
    "aliased_import",
})

_TS_MODULE_CHILD_TYPES = frozenset({
    "string",
    "string_fragment",
})


@dataclass(frozen=True)
class CallerInfo:
    """A changed file imported by the current file, with modified symbol names."""

    changed_path: str
    function_names: Tuple[str, ...]


def _decode_text(node) -> str:
    raw = node.text
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _extract_module_from_python_node(node) -> Optional[str]:
    """Best-effort module path from import_statement / import_from_statement."""
    parts: List[str] = []
    for child in node.children:
        if child.type in _PYTHON_MODULE_CHILD_TYPES:
            text = _decode_text(child).strip()
            if text:
                parts.append(text)
        elif child.type == "wildcard_import":
            parts.append("*")
    if not parts:
        return None
    return "".join(parts)


def _extract_module_from_ts_node(node) -> Optional[str]:
    """Extract module string from import_declaration (quoted path)."""
    for child in node.children:
        if child.type in ("string", "string_fragment"):
            text = _decode_text(child).strip().strip("'\"")
            if text:
                return text
    return None


def _walk_imports(node, lang_key: str, out: List[str]) -> None:
    if node.type in _IMPORT_NODE_TYPES:
        if lang_key == "python":
            mod = _extract_module_from_python_node(node)
        elif lang_key in ("typescript", "javascript", "tsx", "jsx"):
            mod = _extract_module_from_ts_node(node)
        else:
            mod = _extract_module_from_python_node(node) or _extract_module_from_ts_node(node)
        if mod:
            out.append(mod)
    for child in node.children:
        _walk_imports(child, lang_key, out)


def extract_imports(path: str, source: str) -> List[str]:
    """Return raw import module strings from a source file via tree-sitter."""
    if not source or not source.strip():
        return []

    lang = get_language_for_path(path, source)
    if lang is None:
        return []

    from src.intelligence.ast.language_registry import detect_language_key

    lang_key = detect_language_key(path, source) or "unknown"
    parser = Parser()
    parser.set_language(lang)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    modules: List[str] = []
    _walk_imports(tree.root_node, lang_key, modules)
    return modules


def _path_stems(path: str) -> Set[str]:
    """Normalized stems for matching import paths to PR file paths."""
    p = Path(path.replace("\\", "/"))
    stems: Set[str] = set()
    name = p.name
    stem = p.stem
    if stem:
        stems.add(stem.lower())
    if name:
        stems.add(name.lower())
    # path without extension: src/foo/bar -> foo/bar, bar
    parts = list(p.parts)
    if parts:
        no_ext = "/".join(parts[:-1] + [stem]) if len(parts) > 1 else stem
        stems.add(no_ext.lower().replace("\\", "/"))
        stems.add(parts[-2].lower() + "/" + stem.lower() if len(parts) >= 2 else stem.lower())
    return {s for s in stems if s}


def _import_matches_path(import_str: str, target_path: str, target_stems: Set[str]) -> bool:
    """True if import_str likely refers to target_path within the same PR."""
    imp = import_str.strip().strip("'\"")
    if not imp:
        return False

    imp_norm = imp.replace("\\", "/").lower()
    # strip trailing /index, .ts, .py extensions from import
    for suffix in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py"):
        if imp_norm.endswith(suffix):
            imp_norm = imp_norm[: -len(suffix)]
            break
    if imp_norm.endswith("/index"):
        imp_norm = imp_norm[: -len("/index")]

    target_norm = target_path.replace("\\", "/").lower()
    target_no_ext = str(Path(target_norm).with_suffix(""))
    target_stem = Path(target_path).stem.lower()

    if imp_norm == target_no_ext or imp_norm.endswith("/" + target_stem):
        return True
    if imp_norm.endswith(target_stem) and (
        imp_norm == target_stem or imp_norm.endswith("/" + target_stem)
    ):
        return True

    imp_base = imp_norm.split("/")[-1] if "/" in imp_norm else imp_norm
    if imp_base in target_stems:
        return True
    if imp_base == target_stem:
        return True

    return False


def resolve_pr_import_graph(
    file_chunks: List[Tuple[str, str]],
    path_to_content: Dict[str, str],
    mod_funcs_by_path: Dict[str, List[FunctionChunk]],
) -> Dict[str, List[CallerInfo]]:
    """Map importer path -> list of changed files it imports (PR-internal only)."""
    changed_paths = [p for p, fd in file_chunks if fd.strip()]
    if len(changed_paths) < 2:
        return {}

    path_to_stems: Dict[str, Set[str]] = {p: _path_stems(p) for p in changed_paths}

    graph: Dict[str, List[CallerInfo]] = {}

    for importer_path in changed_paths:
        content = path_to_content.get(importer_path) or ""
        if not content.strip():
            continue
        try:
            imports = extract_imports(importer_path, content)
        except Exception as e:
            logger.debug("extract_imports failed for %s: %s", importer_path, e)
            continue

        callers: List[CallerInfo] = []
        for changed_path in changed_paths:
            if changed_path == importer_path:
                continue
            stems = path_to_stems.get(changed_path, set())
            if not any(_import_matches_path(imp, changed_path, stems) for imp in imports):
                continue
            funcs = mod_funcs_by_path.get(changed_path) or []
            names = tuple(
                sorted({f.name for f in funcs if f.name})
            )
            callers.append(CallerInfo(changed_path=changed_path, function_names=names))

        if callers:
            graph[importer_path] = callers

    return graph


__all__ = [
    "CallerInfo",
    "extract_imports",
    "resolve_pr_import_graph",
]
