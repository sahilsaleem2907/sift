"""Language-analyzer seam: pluggable, deterministic AST verdict analyzers.

Each language provides a `LanguageAnalyzer` that inspects a parsed file and emits
high-precision, deterministic findings for bug classes a type-checker is weak on
(e.g. dataclass mutable defaults). Findings are shaped like Semgrep/pyright dicts
(`{line, message, severity, check_id}`) so they ride the existing
`promote_static_findings` path as `critic_exempt` — the guaranteed floor.

Providers are keyed off the tree-sitter `language_registry` key. A language with no
registered provider degrades gracefully (no verdicts ⇒ the finding, if any, falls
back to the LLM/reasoning path).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple


@dataclass(frozen=True)
class AnalyzerFinding:
    """A deterministic verdict, shaped to flow through promote_static_findings."""

    line: int
    message: str
    check_id: str
    severity: str = "ERROR"

    def as_dict(self) -> dict:
        return {
            "line": self.line,
            "message": self.message,
            "severity": self.severity,
            "check_id": self.check_id,
        }


class LanguageAnalyzer(Protocol):
    """A per-language provider of deterministic verdict analyzers."""

    lang_key: str

    def analyze(
        self, path: str, source: str, changed_ranges: List[Tuple[int, int]]
    ) -> List[AnalyzerFinding]:
        """Return findings whose location overlaps the PR's changed line ranges."""
        ...
