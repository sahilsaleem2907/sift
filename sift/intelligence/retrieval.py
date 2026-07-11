"""Effort-scaled context assembly for per-file review (Phase 4)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from sift.intelligence.ast.diff_ast import get_new_file_plus_line_ranges
from sift.intelligence.ast.function_extract import FunctionChunk
from sift.intelligence.capability import ModelCapability
from sift.intelligence.effort import EffortPlan
from sift.intelligence.llm_client import (
    _format_caller_context,
    _format_codeql_findings,
    _format_linter_issues,
    _format_semgrep_findings,
    _format_similar_snippets,
)
logger = logging.getLogger(__name__)

_DIFF_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
_CALL_NAME_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_CHARS_PER_TOKEN = 4
_BUDGET_FRACTION = 0.6
_MAX_CALLEE_LINES = 5
_MAX_WINDOW_LINES_WHEN_TRIMMED = 10

# Trim priority: first dropped = lowest value
_TRIM_ORDER = (
    "vector_snippets",
    "callee_signatures",
    "caller_context",
    "semantic_before_after",
    "window_content",
)


@dataclass
class FileContext:
    diff: str = ""
    window_content: str = ""
    semantic_before_after: str = ""
    callee_signatures: str = ""
    static_tools: str = ""
    caller_context: str = ""
    vector_snippets: str = ""

    def total_chars(self) -> int:
        return sum(
            len(getattr(self, name) or "")
            for name in (
                "diff",
                "window_content",
                "semantic_before_after",
                "callee_signatures",
                "static_tools",
                "caller_context",
                "vector_snippets",
            )
        )

    def to_pr_context_dict(self) -> dict[str, Any]:
        """Keys merged into file pr_context for review_file / agentic."""
        out: dict[str, Any] = {}
        if self.semantic_before_after:
            out["semantic_before_after"] = self.semantic_before_after
        if self.callee_signatures:
            out["callee_signatures"] = self.callee_signatures
        if self.static_tools:
            out["retrieval_static_tools"] = self.static_tools
        return out

    def agentic_context_block(self) -> str:
        """Compact context for the agentic loop initial user message."""
        parts = []
        if self.semantic_before_after:
            parts.append(
                "Semantic before/after of changed functions:\n" + self.semantic_before_after
            )
        if self.callee_signatures:
            parts.append(
                "Callee definitions from other PR files:\n" + self.callee_signatures
            )
        if self.caller_context:
            parts.append(self.caller_context)
        if self.static_tools:
            parts.append(self.static_tools)
        return "\n\n---\n\n".join(parts)


def _format_window_content(path: str, content: str, ranges: list[tuple[int, int]]) -> str:
    if not content or not ranges:
        return ""
    lines = content.splitlines()
    out_lines = [
        "Surrounding context (read-only, for understanding the change):",
        f"File: {path}",
    ]
    for start, end in ranges:
        if start > len(lines) or end < 1:
            continue
        out_lines.append(f"Lines {start}–{end}:")
        for i in range(max(0, start - 1), min(len(lines), end)):
            out_lines.append(f"  {i + 1:4d} | {lines[i]}")
        out_lines.append("")
    return "\n".join(out_lines).strip()


def _window_ranges(
    path: str,
    file_diff: str,
    content: str,
    mod_funcs: list[FunctionChunk],
) -> list[tuple[int, int]]:
    if mod_funcs:
        return [(c.start_line, c.end_line) for c in mod_funcs]
    plus_ranges = get_new_file_plus_line_ranges(file_diff)
    if not plus_ranges:
        return []
    num_lines = len(content.splitlines()) if content else 0
    if not num_lines:
        return plus_ranges
    return [
        (max(1, s - 20), min(num_lines, e + 20))
        for s, e in plus_ranges
    ]


def _old_lines_for_new_range(
    file_diff: str, new_start: int, new_end: int
) -> list[str]:
    """Collect `-` lines from hunks where a `+` line falls in [new_start, new_end]."""
    new_line = 0
    old_line = 0
    pending_old: list[str] = []
    collected: list[str] = []

    for line in file_diff.splitlines():
        m = _DIFF_HUNK_RE.match(line)
        if m:
            old_line = int(m.group(1))
            new_line = int(m.group(3))
            pending_old = []
            continue
        if line.startswith(("diff ", "--- ", "+++ ", "index ")):
            continue
        if not line:
            continue
        prefix = line[0]
        if prefix == " ":
            old_line += 1
            new_line += 1
        elif prefix == "-":
            pending_old.append(line[1:])
            old_line += 1
        elif prefix == "+":
            if new_start <= new_line <= new_end and pending_old:
                collected.extend(pending_old)
                pending_old = []
            new_line += 1
        else:
            continue

    return collected


def _semantic_before_after(
    path: str, file_diff: str, mod_funcs: list[FunctionChunk]
) -> str:
    if not mod_funcs:
        return ""
    blocks: list[str] = []
    for fc in mod_funcs:
        name = fc.name or "?"
        old_lines = _old_lines_for_new_range(file_diff, fc.start_line, fc.end_line)
        old_body = "\n".join(old_lines).strip() if old_lines else "(new or no removed lines)"
        new_body = (fc.text or "").strip()
        blocks.append(
            f"--- old: {name} ({path}) ---\n{old_body}\n"
            f"+++ new: {name} ({path}) ---\n{new_body}"
        )
    return "\n\n".join(blocks)


def _call_names_from_added_lines(file_diff: str) -> set[str]:
    names: set[str] = set()
    for line in file_diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        text = line[1:]
        for m in _CALL_NAME_RE.finditer(text):
            name = m.group(1)
            if name not in ("if", "for", "while", "return", "def", "class", "import", "print"):
                names.add(name)
    return names


def _callee_signatures(
    path: str,
    file_diff: str,
    mod_funcs_by_path: dict[str, list[FunctionChunk]],
) -> str:
    call_names = _call_names_from_added_lines(file_diff)
    if not call_names:
        return ""

    blocks: list[str] = []
    for other_path, funcs in mod_funcs_by_path.items():
        if other_path == path:
            continue
        for fc in funcs or []:
            if fc.name and fc.name in call_names:
                lines = (fc.text or "").splitlines()[:_MAX_CALLEE_LINES]
                snippet = "\n".join(lines)
                blocks.append(f"--- {other_path} :: {fc.name} ---\n{snippet}")
                call_names.discard(fc.name)

    return "\n\n".join(blocks)


def _assemble_static_tools(pr_context: dict[str, Any]) -> str:
    parts: list[str] = []
    semgrep = pr_context.get("semgrep_findings") or []
    block = _format_semgrep_findings(semgrep)
    if block:
        parts.append(block)
    codeql = pr_context.get("codeql_findings") or []
    block = _format_codeql_findings(codeql)
    if block:
        parts.append(block)
    linter = pr_context.get("linter_issues") or []
    block = _format_linter_issues(linter)
    if block:
        parts.append(block)
    return "\n\n".join(parts)


def build_context(
    path: str,
    file_diff: str,
    pr_context: Optional[dict[str, Any]],
    plan: EffortPlan,
    cap: ModelCapability,
    path_to_content: dict[str, str],
    mod_funcs_by_path: dict[str, list[FunctionChunk]],
    import_graph: Optional[dict[str, list]] = None,
) -> FileContext:
    """Assemble context layers for a file review based on effort depth."""
    _ = import_graph
    file_diff = file_diff or ""
    pr_context = pr_context or {}
    content = path_to_content.get(path) or ""
    mod_funcs = list(mod_funcs_by_path.get(path) or [])

    # Prefer ranges already computed in review_engine file_context
    fc = pr_context.get("file_context") or {}
    ranges = fc.get("ranges") if isinstance(fc, dict) else None
    if not ranges:
        ranges = _window_ranges(path, file_diff, content, mod_funcs)

    ctx = FileContext(diff=file_diff)
    ctx.window_content = _format_window_content(path, content, ranges or [])

    caller_infos = pr_context.get("caller_context")
    if caller_infos:
        ctx.caller_context = _format_caller_context(caller_infos)

    similar = pr_context.get("similar_snippets")
    if similar:
        ctx.vector_snippets = _format_similar_snippets(similar)

    ctx.static_tools = _assemble_static_tools(pr_context)

    depth = plan.context_depth
    if depth >= 1:
        ctx.semantic_before_after = _semantic_before_after(path, file_diff, mod_funcs)
    if depth >= 2:
        ctx.callee_signatures = _callee_signatures(
            path, file_diff, mod_funcs_by_path
        )

    budget_chars = int(cap.context_window * _BUDGET_FRACTION * _CHARS_PER_TOKEN)
    return trim_to_budget(ctx, budget_chars)


def _total_chars(
    diff: str,
    static_tools: str,
    fields: dict[str, str],
) -> int:
    return len(diff) + len(static_tools) + sum(len(v) for v in fields.values())


def trim_to_budget(ctx: FileContext, budget_chars: int) -> FileContext:
    """Drop lowest-priority blocks until within budget; diff is never dropped."""
    fields = {
        name: getattr(ctx, name) or ""
        for name in _TRIM_ORDER
    }

    def total() -> int:
        return _total_chars(ctx.diff, ctx.static_tools, fields)

    if total() <= budget_chars:
        return ctx

    for name in _TRIM_ORDER:
        if total() <= budget_chars:
            break
        fields[name] = ""

    while total() > budget_chars and fields["window_content"]:
        lines = fields["window_content"].splitlines()
        if len(lines) <= 4:
            fields["window_content"] = ""
            break
        fields["window_content"] = "\n".join(
            lines[:_MAX_WINDOW_LINES_WHEN_TRIMMED] + ["  ... (truncated for context budget)"]
        )

    return FileContext(
        diff=ctx.diff,
        window_content=fields["window_content"],
        semantic_before_after=fields["semantic_before_after"],
        callee_signatures=fields["callee_signatures"],
        static_tools=ctx.static_tools,
        caller_context=fields["caller_context"],
        vector_snippets=fields["vector_snippets"],
    )
