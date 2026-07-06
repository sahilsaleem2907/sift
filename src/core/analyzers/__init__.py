"""Deterministic verdict-analyzer registry (language seam).

`run_analyzers` picks the provider for a file's language, runs it against the PR's
changed line ranges, and returns Semgrep/pyright-shaped finding dicts ready for
`promote_static_findings`. Languages without a provider return `[]` (graceful
degradation), so the pipeline behaves identically where no analyzer exists.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from src.intelligence.ast.diff_ast import get_new_file_plus_line_ranges
from src.intelligence.ast.language_registry import detect_language_key

from .base import LanguageAnalyzer
from .python_rules import PythonAnalyzer

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, LanguageAnalyzer] = {}


def register(analyzer: LanguageAnalyzer) -> None:
    _REGISTRY[analyzer.lang_key] = analyzer


register(PythonAnalyzer())


def run_analyzers(path: str, source: str, diff_chunk: str) -> List[dict]:
    """Return deterministic verdict findings (as dicts) for the changed lines of `path`.

    Returns [] when the language has no registered analyzer, the file is empty, or
    the diff has no added-line ranges.
    """
    if not source or not diff_chunk:
        return []
    key = detect_language_key(path, source)
    if not key:
        return []
    analyzer = _REGISTRY.get(key)
    if analyzer is None:
        return []
    ranges = get_new_file_plus_line_ranges(diff_chunk)
    if not ranges:
        return []
    try:
        findings = analyzer.analyze(path, source, ranges)
    except Exception as e:  # analyzers must never break a review
        logger.warning("[analyzer] %s failed on %s: %s", key, path, e)
        return []
    return [f.as_dict() for f in findings]


__all__ = ["run_analyzers", "register"]
