"""Local Ollama integration for code review."""
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from src import config
from src.intelligence.ast.diff_ast import get_new_file_plus_line_ranges

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a concise code reviewer. Given a git diff, provide a short professional review:
- A few bullet points on correctness, style, and possible improvements.
- Be brief and actionable. Do not repeat the diff."""

REVIEW_FILE_SYSTEM = """You are a code reviewer. Given a single file's unified diff, write your review in plain text or markdown.

CRITICAL - Understanding the diff:
- Lines starting with "-" are OLD (removed); do NOT cite these line numbers in your review.
- Lines starting with "+" are NEW (added/changed); these are the only lines you should reference.
- When you mention a line number, it must be the line number in the NEW file (the right side), i.e. a "+" line. Never cite line numbers from "-" (removed) lines.

FORMAT - So we can attach your comments to the right line, you MUST start each comment with exactly "Line N:" where N is the new-file line number (e.g. "Line 11:"). Then write your issue description and optional suggested fix. Example:
Line 11: Consider adding a null check here. **Suggested fix:** ...
"""

SUMMARIZE_SYSTEM = """Summarize the following inline review comments in a few sentences or bullet points for a pull request Conversation tab. Be brief and professional."""


async def _call_ollama(system: str, user_content: str) -> str:
    """Call Ollama chat API; return assistant message content or empty string."""
    base_url = config.OLLAMA_BASE_URL
    model = config.OLLAMA_MODEL
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    payload = {"model": model, "messages": messages, "stream": False}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        logger.error("Ollama request failed: %s", e)
        raise
    msg = data.get("message")
    if not msg or "content" not in msg:
        logger.warning("Unexpected Ollama response shape: %s", data)
        return ""
    return (msg["content"] or "").strip()


# Match line-number references so we can parse LLM output. Supports:
# "Line 42:", "line 42", "L42", "#42", "at line 42", "on line 42", "Line 42 -", "Line 42."
_LINE_REF_RE = re.compile(
    r"(?:^|\n)\s*(?:(?:Line|line|L)\s*|#\s*|(?:at|on)\s+line\s+)(\d+)\s*[:\.\-]?\s*",
    re.IGNORECASE,
)


def _normalize_comment_body(body: str) -> str:
    """Ensure body has **Suggested fix:** before code blocks and consistent formatting."""
    if not body or not body.strip():
        return body
    text = body.strip()
    # If there's a fenced code block but no "Suggested fix" / "Optimal solution" already in text, add one
    if re.search(r"```", text) and not re.search(
        r"Suggested fix|Optimal solution", text, re.IGNORECASE
    ):
        text = re.sub(r"(\s*)```", r"\1**Suggested fix:**\n\n```", text, count=1)
    return text


def _parse_review_file_response(raw: str, path: str) -> List[Dict[str, Any]]:
    """Parse freeform LLM response into list of {line, body}. Extracts line numbers from text."""
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    matches = list(_LINE_REF_RE.finditer(text))
    if not matches:
        logger.debug("No line references found in review for %s", path)
        return []

    out: List[Dict[str, Any]] = []
    for i, m in enumerate(matches):
        try:
            line_int = int(m.group(1))
            if line_int <= 0:
                continue
        except (TypeError, ValueError):
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        body = _normalize_comment_body(body)
        out.append({"line": line_int, "body": body})
    return out


def _format_semgrep_findings(findings: List[Dict[str, Any]]) -> str:
    """Format Semgrep findings for inclusion in the LLM prompt."""
    if not findings:
        return ""
    lines = []
    for f in findings:
        line = f.get("line", "?")
        msg = (f.get("message") or "").strip()
        severity = f.get("severity") or "WARNING"
        rule_id = f.get("check_id") or ""
        lines.append(f"Line {line}: [{rule_id}] {msg} (severity: {severity})")
    return "Semgrep findings for this file (consider in your review):\n" + "\n".join(lines)


def _format_ast_diff(ast_diff: Dict[str, Any]) -> str:
    """Format diff-aware AST metadata into a compact, LLM-friendly text block.

    The goal is to surface structural context (node types, spans, and texts) for
    only the changed '+' lines, without overwhelming the model with raw JSON.
    """
    out_lines: List[str] = []

    path = ast_diff.get("path", "?")
    lang = ast_diff.get("lang", "?")
    out_lines.append(f"File: {path} (language: {lang})")

    changed_ranges = ast_diff.get("changed_ranges") or []
    if changed_ranges:
        ranges_str = ", ".join(
            f"[lines {r.get('start_line')}–{r.get('end_line')}]" for r in changed_ranges
        )
        out_lines.append(f"Changed '+' line ranges: {ranges_str}")

    nodes = ast_diff.get("nodes") or []
    if not nodes:
        return "\n".join(out_lines)

    out_lines.append("AST nodes on changed '+' lines (one per line below):")
    for n in nodes:
        line = n.get("start_line")
        start_col = n.get("start_col")
        end_col = n.get("end_col")
        node_type = n.get("type") or "?"
        text = (n.get("text") or "").replace("\n", "\\n")
        out_lines.append(
            f"- line {line}, cols {start_col}-{end_col}, type={node_type}, text={text!r}"
        )

    return "\n".join(out_lines)

def _format_linter_issues(issues: List[Dict[str, Any]]) -> str:
    """Format linter issues (with snippets) for inclusion in the LLM prompt. Only lines on the diff."""
    if not issues:
        return ""
    lines = []
    for i in issues:
        line = i.get("line", "?")
        source = i.get("source") or "linter"
        rule_id = (i.get("rule_id") or "").strip()
        msg = (i.get("message") or "").strip()
        snippet = (i.get("snippet") or "").strip()
        part = f"Line {line} [{source}"
        if rule_id:
            part += f" / {rule_id}"
        part += f"]: {msg}"
        if snippet:
            part += f"\n  Code:   {snippet}"
        lines.append(part)
    return "Linter issues on changed lines (consider in your review):\n" + "\n".join(lines)


async def review_file(
    diff_chunk: str,
    path: str,
    pr_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Review a single file's diff; return list of {line, body} for inline comments.

    path is for logging; we do not rely on LLM for path. Line is the line number in the new file (right side).
    """
    user_parts = []
    if pr_context:
        semgrep_findings = pr_context.get("semgrep_findings") or []
        semgrep_block = _format_semgrep_findings(semgrep_findings)
        if semgrep_block:
            user_parts.append(semgrep_block)
        ast_diff = pr_context.get("ast_diff")
        if ast_diff:
            try:
                user_parts.append(
                    "Structured AST metadata from tree-sitter for this file, "
                    "restricted to nodes whose line ranges fall fully within new '+' diff lines.\n"
                    "Use this as structural context, not as full source code. "
                    "The following list shows AST nodes on changed lines with their spans and source text:\n"
                    f"{_format_ast_diff(ast_diff)}"
                )
            except TypeError as e:
                logger.debug("Failed to serialize ast_diff for %s: %s", path, e)
        linter_issues = pr_context.get("linter_issues") or []
        linter_block = _format_linter_issues(linter_issues)
        if linter_block:
            user_parts.append(linter_block)
            logger.debug(
                "LLM linter_issues input: file=%s, count=%d, block_preview=%s",
                path,
                len(linter_issues),
                linter_block[:500] + "..." if len(linter_block) > 500 else linter_block,
            )
        title = pr_context.get("title") or ""
        body = pr_context.get("body") or ""
        if title or body:
            user_parts.append(f"PR title: {title}\n\nPR description: {body}")

    diff_intro = (
        f"File: {path}\n\n"
        "Diff (legend: '-' = old/removed, '+' = new/added — cite line numbers only for '+' lines)."
    )
    # Use line numbers derived from the diff itself (where '+' lines are in the new file),
    # so the LLM is told the correct line numbers (e.g. 11 for the changed line, not the hunk start 8).
    ranges = get_new_file_plus_line_ranges(diff_chunk)
    if ranges:
        line_nums = []
        for start, end in ranges:
            if start == end:
                line_nums.append(str(start))
            else:
                line_nums.append(f"{start}-{end}")
        if line_nums:
            diff_intro += (
                f" The '+' lines in the diff below are at new-file line(s): {', '.join(line_nums)}. "
                f"Cite these line numbers using the format 'Line N:' (e.g. Line {line_nums[0].split('-')[0]}:) at the start of each comment."
            )
    diff_intro += "\n\n"
    user_parts.append(diff_intro + diff_chunk)
    user_content = "\n\n---\n\n".join(user_parts)

    semgrep_findings = (pr_context or {}).get("semgrep_findings") or []
    linter_issues = (pr_context or {}).get("linter_issues") or []
    logger.debug(
        "LLM input: file=%s, content_length=%d, semgrep_findings=%d, linter_issues=%d, has_pr_context=%s",
        path,
        len(user_content),
        len(semgrep_findings),
        len(linter_issues),
        bool(pr_context and (pr_context.get("title") or pr_context.get("body"))),
    )
    logger.debug("LLM input full payload for %s:\n%s", path, user_content)

    raw = await _call_ollama(REVIEW_FILE_SYSTEM, user_content)
    logger.debug("LLM raw output for %s:\n%s", path, raw)
    return _parse_review_file_response(raw, path)


async def summarize_review(comments: List[Dict[str, Any]]) -> str:
    """Produce a short summary string for the Conversation tab from the list of comments we're posting.

    comments: list of {path, line, body} (or at least body for each).
    """
    if not comments:
        return "No inline comments for this review."
    lines = []
    for c in comments:
        path = c.get("path", "?")
        line = c.get("line", "?")
        body = (c.get("body") or "").strip()
        if body:
            lines.append(f"- **{path}** (line {line}): {body}")
    user_content = "Inline comments being posted:\n\n" + "\n".join(lines)
    raw = await _call_ollama(SUMMARIZE_SYSTEM, user_content)
    return raw if raw else "Review completed with inline comments on the Files changed tab."


async def review(diff: str, pr_context: Optional[Dict[str, Any]] = None) -> str:
    """Call Ollama to generate a code review for the given diff.

    pr_context may contain "title" and "body" for the PR description.
    Returns the model's review text. Kept for backward compatibility / fallback.
    """
    user_content = diff
    if pr_context:
        title = pr_context.get("title") or ""
        body = pr_context.get("body") or ""
        if title or body:
            user_content = f"PR title: {title}\n\nPR description:\n{body}\n\n---\n\nDiff:\n{diff}"
    raw = await _call_ollama(SYSTEM_PROMPT, user_content)
    return raw if raw else "Review could not be generated."
