import logging
import os
from functools import lru_cache
from typing import Dict, Optional

from tree_sitter import Language
from tree_sitter_languages import get_language


logger = logging.getLogger(__name__)


# Map common file extensions to tree-sitter language identifiers used by
# `tree_sitter_languages.get_language`.
_EXTENSION_TO_LANG_KEY: Dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "tsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "c_sharp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".rs": "rust",
    ".hs": "haskell",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
}


def _detect_from_extension(path: str) -> Optional[str]:
    _, ext = os.path.splitext(path)
    if not ext:
        return None
    lang = _EXTENSION_TO_LANG_KEY.get(ext.lower())
    if lang:
        logger.debug("Detected language %s from extension for %s", lang, path)
    return lang


def _detect_from_shebang(source: Optional[str]) -> Optional[str]:
    if not source:
        return None
    first_line = source.splitlines()[0].strip() if source.splitlines() else ""
    if not first_line.startswith("#!"):
        return None
    # Very small set of common interpreters; extend as needed.
    if "python" in first_line:
        logger.debug("Detected python from shebang: %s", first_line)
        return "python"
    if "node" in first_line or "deno" in first_line:
        logger.debug("Detected javascript from shebang: %s", first_line)
        return "javascript"
    if "bash" in first_line or "sh" in first_line or "zsh" in first_line:
        logger.debug("Detected bash from shebang: %s", first_line)
        return "bash"
    return None


def detect_language_key(path: str, source: Optional[str] = None) -> Optional[str]:
    """Best-effort detection of a tree-sitter language key for a file.

    Prefers extension-based mapping and falls back to simple shebang inspection.
    Returns a stable language key understood by `tree_sitter_languages.get_language`,
    or None if the language cannot be determined.
    """
    key = _detect_from_extension(path)
    if key:
        return key
    shebang_key = _detect_from_shebang(source)
    if shebang_key:
        return shebang_key
    logger.debug("Could not detect language for %s", path)
    return None


@lru_cache(maxsize=None)
def get_language_by_key(lang_key: str) -> Optional[Language]:
    """Return a cached tree-sitter Language for the given key, or None on failure."""
    try:
        lang = get_language(lang_key)
        logger.debug("Loaded tree-sitter language for key %s", lang_key)
        return lang
    except Exception:
        # If the underlying language bundle does not support this key,
        # we gracefully fall back to no AST for this file.
        return None


def get_language_for_path(path: str, source: Optional[str] = None) -> Optional[Language]:
    """Detect the language for a file and return its tree-sitter Language, if available."""
    key = detect_language_key(path, source)
    if not key:
        return None
    lang = get_language_by_key(key)
    if lang is None:
        logger.debug("No tree-sitter language available for key %s (path=%s)", key, path)
    return lang


__all__ = ["detect_language_key", "get_language_by_key", "get_language_for_path"]

