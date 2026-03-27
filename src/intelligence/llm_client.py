"""LLM integration for code review (LiteLLM: OpenAI, Anthropic, Gemini, Ollama, Azure, Bedrock)."""
import json
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from litellm import acompletion

from src import config
from src.intelligence.ast.diff_ast import get_new_file_plus_line_ranges

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a concise code reviewer. Given a git diff, provide a short professional review:
- A few bullet points on correctness, style, and possible improvements.
- Be brief and actionable. Do not repeat the diff."""

REVIEW_FILE_SYSTEM = """You are a code reviewer focused on correctness. Your job is to find real bugs and issues.

Look specifically for:
- Logic errors, wrong conditions, off-by-one errors
- Unhandled None/null dereferences
- Unhandled exceptions or missing error handling
- Security issues (injection, auth bypass, improper validation)
- Resource leaks (unclosed files/connections)
- Type mismatches or wrong API usage

Respond ONLY with a JSON array. No prose, no markdown fences around the array. Each element:
{
  "line": <integer — must be a line number marked [L<n>] in the diff below>,
  "severity": "bug" | "security" | "warning" | "suggestion",
  "title": "<10 words max>",
  "body": "<description of the issue>",
  "fix": "<optional: corrected code only, no diff markers>"
}

Rules:
- "line" MUST be one of the annotated [L<n>] numbers from the diff. Never invent a line number.
- Only report issues on changed lines (marked with +).
- Omit "fix" if no clean fix is obvious.
- Return [] if there is nothing significant to report.
"""

SUMMARIZE_SYSTEM = """Summarize the following inline review comments in a few sentences or bullet points for a pull request Conversation tab. Be brief and professional."""

# Match severity badge at start of comment body (text or shields.io image).
_SUMMARY_SEVERITY_RE = re.compile(r"^\*\*\[(BUG|SECURITY|WARNING|SUGGESTION)\]\*\*\s*(.*)")
_SUMMARY_SHIELD_RE = re.compile(r"^!\[(BUG|SECURITY|WARNING|SUGGESTION)\]\(https://[^)]+\)\s*(.*)", re.DOTALL)

# Hunk header for unified diff: @@ -old_start[,old_count] +new_start[,new_count] @@
_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def _annotate_diff_with_line_numbers(diff_chunk: str, path: str) -> str:
    """Append [L<n>] to each '+' line in the diff so the LLM can cite exact new-file line numbers."""
    if not diff_chunk or not diff_chunk.strip():
        return diff_chunk
    comment_marker = "  # " if (path.endswith(".py") or ".py/" in path) else "  // "
    lines_out: List[str] = []
    current_new_line: Optional[int] = None
    for line in diff_chunk.splitlines():
        m = _DIFF_HUNK_RE.match(line)
        if m:
            current_new_line = int(m.group(1))
            lines_out.append(line)
            continue
        if current_new_line is None:
            lines_out.append(line)
            continue
        if not line:
            lines_out.append(line)
            continue
        prefix = line[0]
        if prefix == " ":
            current_new_line += 1
            lines_out.append(line)
        elif prefix == "+" and not line.startswith("+++"):
            lines_out.append(line + comment_marker + f"[L{current_new_line}]")
            current_new_line += 1
        elif prefix == "-" and not line.startswith("---"):
            lines_out.append(line)
        else:
            lines_out.append(line)
    return "\n".join(lines_out)


async def _call_llm(system: str, user_content: str) -> str:
    """Call configured LLM provider via LiteLLM; return assistant message content or empty string."""
    response = await acompletion(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        api_base=config.LLM_API_BASE or None,
        timeout=120.0,
    )
    return (response.choices[0].message.content or "").strip()


# Match line-number references so we can parse LLM output. Supports:
# "Line 42:", "line 42", "L42", "#42", "at line 42", "on line 42", "Line 42 -", "Line 42."
_LINE_REF_RE = re.compile(
    r"(?:^|\n)\s*(?:(?:Line|line|L)\s*|#\s*|(?:at|on)\s+line\s+)(\d+)\s*[:\.\-]?\s*",
    re.IGNORECASE,
)


def _strip_diff_markers_from_code_block(content: str) -> str:
    """Strip diff markers (+, -, @@) from code block content so it is copy-pasteable."""
    lines = content.split("\n")
    out = []
    for line in lines:
        # Remove @@ hunk headers (e.g. @@ -1,3 +1,4 @@)
        if re.match(r"^\s*@@\s.*\s@@\s*$", line) or re.match(r"^\s*@@\s", line):
            continue
        # Strip leading "+ " or "- " (diff added/removed line markers)
        if line.startswith("+ "):
            line = line[2:]
        elif line.startswith("- "):
            line = line[2:]
        elif line.startswith("+") and len(line) > 1 and line[1] != "+":
            line = line[1:]
        elif line.startswith("-") and len(line) > 1 and line[1] != "-":
            line = line[1:]
        out.append(line)
    return "\n".join(out)


def _normalize_comment_body(body: str) -> str:
    """Ensure body has **Suggested fix:** before code blocks, consistent formatting, and clean copy-pasteable code (no diff markers)."""
    if not body or not body.strip():
        return body
    text = body.strip()
    # Strip diff markers from inside fenced code blocks (safety net if LLM emitted diff syntax)
    def replace_code_block(m: re.Match) -> str:
        fence = m.group(1)  # opening ``` optionally with lang
        inner = m.group(2)  # content
        return fence + "\n" + _strip_diff_markers_from_code_block(inner) + "\n```"
    text = re.sub(r"(```[\w]*)\n(.*?)```", replace_code_block, text, flags=re.DOTALL)
    # If there's a fenced code block but no "Suggested fix" / "Optimal solution" already in text, add one
    if re.search(r"```", text) and not re.search(
        r"Suggested fix|Optimal solution", text, re.IGNORECASE
    ):
        text = re.sub(r"(\s*)```", r"\1**Suggested fix:**\n\n```", text, count=1)
    return text


# Severity badge markdown (shields.io) for inline comments and summary. Order: bug, security, warning, suggestion.
_BADGE_STYLE = "for-the-badge"
_SEV_META = [
    ("bug", f"![BUG](https://img.shields.io/badge/BUG-FF4444?style={_BADGE_STYLE})"),
    ("security", f"![SECURITY](https://img.shields.io/badge/SECURITY-FF8C00?style={_BADGE_STYLE})"),
    ("warning", f"![WARNING](https://img.shields.io/badge/WARNING-FFD700?style={_BADGE_STYLE})"),
    ("suggestion", f"![SUGGESTION](https://img.shields.io/badge/SUGGESTION-4A90D9?style={_BADGE_STYLE})"),
]
_SEV_BADGE_BY_KEY = {key: badge for key, badge in _SEV_META}

# Summary-only: shields path label-messageColor with dark labelColor (left) and lighter message (right).
_SEV_SUMMARY_SPEC: Dict[str, Dict[str, str]] = {
    "bug": {"label": "BUG", "message_color": "FF4444", "label_color": "AA0000"},
    "security": {"label": "SECURITY", "message_color": "FF8C00", "label_color": "CC5500"},
    "warning": {"label": "WARNING", "message_color": "FFD700", "label_color": "B8860B"},
    "suggestion": {"label": "SUGGESTION", "message_color": "4A90D9", "label_color": "2E5A8A"},
}


def _summary_count_badge_markdown(severity_key: str, count: int) -> str:
    """Shields.io badge: dark label (labelColor) + count (lighter message color). Summary comment only."""
    spec = _SEV_SUMMARY_SPEC.get(severity_key, _SEV_SUMMARY_SPEC["suggestion"])
    label = spec["label"]
    msg_color = spec["message_color"]
    lbl_color = spec["label_color"]
    # Path segments may need encoding for shields (hyphens separate label / message / color).
    seg_label = quote(label, safe="")
    seg_msg = str(int(count))
    seg_color = quote(msg_color, safe="")
    url = (
        f"https://img.shields.io/badge/{seg_label}-{seg_msg}-{seg_color}"
        f"?style={_BADGE_STYLE}&labelColor={lbl_color}"
    )
    return f"![{label} {count}]({url})"


def _format_structured_comment_body(item: Dict[str, Any]) -> str:
    """Turn a parsed JSON review item into a consistent GitHub markdown comment."""
    severity = (item.get("severity") or "suggestion").lower()
    badge = _SEV_BADGE_BY_KEY.get(severity, _SEV_BADGE_BY_KEY["suggestion"])
    title = (item.get("title") or "").strip() or "Issue"
    body_text = (item.get("body") or "").strip()
    fix = (item.get("fix") or "").strip()
    parts = [f"{badge} {title}"]
    if body_text:
        parts.append("\n\n" + body_text)
    if fix:
        fix_clean = _strip_diff_markers_from_code_block(fix)
        parts.append("\n\n**Suggested fix:**\n\n```\n" + fix_clean + "\n```")
    return "".join(parts)


def _extract_json_array(raw: str) -> Optional[List[Any]]:
    """Extract a JSON array from raw LLM output (may be wrapped in prose or markdown)."""
    text = raw.strip()
    # Find first '[' and last ']' to get the array slice
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    end = -1
    in_string = None
    i = start
    while i < len(text):
        c = text[i]
        if in_string:
            if c == "\\" and i + 1 < len(text):
                i += 2
                continue
            if c == in_string:
                in_string = None
            i += 1
            continue
        if c in ('"', "'"):
            in_string = c
            i += 1
            continue
        if c == "[":
            depth += 1
            i += 1
            continue
        if c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
            i += 1
            continue
        i += 1
    if end < 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _parse_review_file_response(raw: str, path: str) -> List[Dict[str, Any]]:
    """Parse LLM response: prefer JSON array; fall back to freeform line-ref regex."""
    if not raw or not raw.strip():
        return []
    text = raw.strip()

    # Try structured JSON first
    arr = _extract_json_array(text)
    if arr is not None and isinstance(arr, list):
        out: List[Dict[str, Any]] = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            try:
                line_val = item.get("line")
                line_int = int(line_val) if line_val is not None else None
            except (TypeError, ValueError):
                continue
            if line_int is None or line_int <= 0:
                continue
            body = _format_structured_comment_body(item)
            if body.strip():
                out.append({"line": line_int, "body": body})
        if out:
            return out
        logger.debug("JSON array empty or invalid items for %s", path)

    # Fallback: freeform "Line N:" parsing
    matches = list(_LINE_REF_RE.finditer(text))
    if not matches:
        logger.debug("No line references found in review for %s", path)
        return []
    out = []
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


def _format_file_context(file_context: Dict[str, Any]) -> str:
    """Format surrounding file context (read-only) for the LLM: path, line ranges, and numbered lines."""
    path = file_context.get("path") or "?"
    content = (file_context.get("content") or "").strip()
    ranges = file_context.get("ranges") or []
    if not content or not ranges:
        return ""
    lines = content.splitlines()
    out_lines: List[str] = [
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
        suffix = " [FILE-WIDE]" if f.get("critical_bypass") else ""
        lines.append(f"Line {line}: [{rule_id}] {msg} (severity: {severity}){suffix}")
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
        if i.get("critical_bypass"):
            part += " [FILE-WIDE]"
        if snippet:
            part += f"\n  Code:   {snippet}"
        lines.append(part)
    return "Linter issues on changed lines (consider in your review):\n" + "\n".join(lines)


def _format_codeql_findings(findings: List[Dict[str, Any]]) -> str:
    """Format CodeQL findings for inclusion in the LLM prompt (same shape as Semgrep)."""
    if not findings:
        return ""
    lines = []
    for f in findings:
        line = f.get("line", "?")
        msg = (f.get("message") or "").strip()
        severity = f.get("severity") or "WARNING"
        rule_id = f.get("check_id") or ""
        suffix = " [FILE-WIDE]" if f.get("critical_bypass") else ""
        lines.append(f"Line {line}: [{rule_id}] {msg} (severity: {severity}){suffix}")
    return "CodeQL findings for this file (consider in your review):\n" + "\n".join(lines)


def _format_similar_snippets(matches: List[Any]) -> str:
    """Format vector-similarity matches into an LLM-friendly context block."""
    if not matches:
        return ""
    _MAX_SNIPPET_LINES = 30
    lines: List[str] = [
        "Similar code elsewhere in this repo (consider for duplication, consistency, or repeated mistakes):"
    ]
    for m in matches:
        header = f"- {m.path} (lines {m.start_line}-{m.end_line})"
        if m.func_name:
            header += f", function `{m.func_name}`"
        snippet = m.chunk_text
        snippet_lines = snippet.splitlines()
        if len(snippet_lines) > _MAX_SNIPPET_LINES:
            snippet = "\n".join(snippet_lines[:_MAX_SNIPPET_LINES]) + "\n... (truncated)"
        lines.append(f"{header}\n```\n{snippet}\n```")
    return "\n".join(lines)


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
        file_context = pr_context.get("file_context")
        if file_context:
            fc_block = _format_file_context(file_context)
            if fc_block:
                user_parts.append(fc_block)
        semgrep_findings = pr_context.get("semgrep_findings") or []
        semgrep_block = _format_semgrep_findings(semgrep_findings)
        if semgrep_block:
            user_parts.append(semgrep_block)
        codeql_findings = pr_context.get("codeql_findings") or []
        codeql_block = _format_codeql_findings(codeql_findings)
        if codeql_block:
            user_parts.append(codeql_block)
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
        "Diff (legend: '-' = old/removed, '+' = new/added). Each added line is annotated with [L<n>] — use that integer as \"line\" in your JSON."
    )
    diff_intro += "\n\n"
    annotated_diff = _annotate_diff_with_line_numbers(diff_chunk, path)
    user_parts.append(diff_intro + annotated_diff)

    similar_snippets = (pr_context or {}).get("similar_snippets")
    if similar_snippets:
        sim_block = _format_similar_snippets(similar_snippets)
        if sim_block:
            user_parts.append(sim_block)
            logger.debug(
                "[Vector] path=%s: appended similar_snippets block to prompt (%d match(es), %d chars)",
                path, len(similar_snippets), len(sim_block),
            )

    user_content = "\n\n---\n\n".join(user_parts)

    semgrep_findings = (pr_context or {}).get("semgrep_findings") or []
    linter_issues = (pr_context or {}).get("linter_issues") or []
    codeql_findings = (pr_context or {}).get("codeql_findings") or []
    logger.debug(
        "LLM input: file=%s, content_length=%d, semgrep_findings=%d, linter_issues=%d, codeql_findings=%d, has_pr_context=%s",
        path,
        len(user_content),
        len(semgrep_findings),
        len(linter_issues),
        len(codeql_findings),
        bool(pr_context and (pr_context.get("title") or pr_context.get("body"))),
    )
    if codeql_findings:
        logger.debug("CodeQL findings for %s: %s", path, codeql_findings)
    logger.debug("LLM input full payload for %s:\n%s", path, user_content)

    raw = await _call_llm(REVIEW_FILE_SYSTEM, user_content)
    logger.debug("LLM raw output for %s:\n%s", path, raw)
    return _parse_review_file_response(raw, path)


def _summary_severity_and_title(body: str) -> tuple:
    """Extract (severity_key, title) from comment body. severity_key is lowercase for counting."""
    if not body or not body.strip():
        return ("suggestion", "")
    text = body.strip()
    m = _SUMMARY_SHIELD_RE.match(text)
    if m:
        return (m.group(1).lower(), (m.group(2) or "").strip().split("\n")[0].strip())
    m = _SUMMARY_SEVERITY_RE.match(text)
    if m:
        return (m.group(1).lower(), (m.group(2) or "").strip().split("\n")[0].strip())
    # Merged comments: "**Issues:**\n- badge ..." or first shield in body
    if "**Issues:**" in text or "!(" in text:
        m2 = _SUMMARY_SHIELD_RE.search(text) or _SUMMARY_SEVERITY_RE.search(text)
        if m2:
            return (m2.group(1).lower(), (m2.group(2) or "").strip().split("\n")[0].strip())
    return ("suggestion", text.split("\n")[0].strip() if text else "")


def _build_structured_summary(comments: List[Dict[str, Any]]) -> str:
    """Build a structured summary with alert blocks and collapsible file details."""
    if not comments:
        return "No inline comments for this review."

    counts: Dict[str, int] = {"bug": 0, "security": 0, "warning": 0, "suggestion": 0}
    by_path: Dict[str, List[Dict[str, Any]]] = {}
    for c in comments:
        path = c.get("path", "?")
        line = c.get("line", "?")
        body = (c.get("body") or "").strip()
        if not body:
            continue
        sev, title = _summary_severity_and_title(body)
        counts[sev] = counts.get(sev, 0) + 1
        by_path.setdefault(path, []).append({
            "line": line,
            "sev": sev,
            "title": title or "(see inline)",
            "body": body,
        })

    total = sum(counts.values())
    num_files = len(by_path)
    lines: List[str] = [
        "## Sift Review",
        "",
        f"> {total} issue(s) found across {num_files} file(s)",
        "",
    ]

    bug_count = counts.get("bug", 0)
    security_count = counts.get("security", 0)
    warning_count = counts.get("warning", 0)
    suggestion_count = counts.get("suggestion", 0)

    if bug_count > 0 or security_count > 0:
        parts: List[str] = []
        if bug_count > 0:
            parts.append(_summary_count_badge_markdown("bug", bug_count))
        if security_count > 0:
            parts.append(_summary_count_badge_markdown("security", security_count))
        blocking_count = bug_count + security_count
        lines.extend(
            [
                "> [!CAUTION]",
                f"> {' '.join(parts)} - **{blocking_count}** blocking issue(s). Do not merge.",
                "",
            ]
        )

    if warning_count > 0:
        lines.extend(
            [
                "> [!WARNING]",
                f"> {_summary_count_badge_markdown('warning', warning_count)} warning issue(s) require attention before merging.",
                "",
            ]
        )

    if suggestion_count > 0:
        lines.extend(
            [
                "> [!TIP]",
                f"> {_summary_count_badge_markdown('suggestion', suggestion_count)} suggestion issue(s) available in the Files changed tab.",
                "",
            ]
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("<details>")
    lines.append(f"<summary>{total} issue(s) across {num_files} file(s) - full breakdown</summary>")
    lines.append("")
    for path in sorted(by_path.keys()):
        items = sorted(by_path[path], key=lambda x: x["line"])
        lines.append(f"### `{path}`")
        lines.append("| Line | Severity | Issue |")
        lines.append("|------|----------|-------|")
        for item in items:
            sev_key = item["sev"] if item["sev"] in _SEV_SUMMARY_SPEC else "suggestion"
            badge = _summary_count_badge_markdown(sev_key, 1)
            title_safe = (item["title"] or "").replace("|", ", ").replace("\n", " ")
            lines.append(f"| {item['line']} | {badge} | {title_safe} |")
        lines.append("")
    lines.append("</details>")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Inline comments with details and suggested fixes are on the Files changed tab.*")
    return "\n".join(lines)


async def summarize_review(comments: List[Dict[str, Any]]) -> str:
    """Produce a structured summary for the Conversation tab: status counts and comments by file.

    comments: list of {path, line, body} (or at least body for each).
    """
    return _build_structured_summary(comments)


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
    raw = await _call_llm(SYSTEM_PROMPT, user_content)
    return raw if raw else "Review could not be generated."
