"""LLM integration for code review (LiteLLM: OpenAI, Anthropic, Gemini, Ollama, Azure, Bedrock)."""
import json
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx
import litellm
from litellm import acompletion

litellm.suppress_debug_info = True

from sift import config
from sift.intelligence.ast.diff_ast import get_new_file_plus_line_ranges
from sift.intelligence.prompts import REVIEW_FILE_SYSTEM, TEST_FILE_APPENDIX

logger = logging.getLogger(__name__)

# Match severity badge at start of comment body (text or shields.io image).
_SUMMARY_SEVERITY_RE = re.compile(
    r"^\*\*\[(BUG|SECURITY|WARNING|SUGGESTION|INFORMATIONAL)\]\*\*\s*(.*)"
)
_SUMMARY_SHIELD_RE = re.compile(
    r"^!\[(BUG|SECURITY|WARNING|SUGGESTION|INFORMATIONAL)\]\(https://[^)]+\)\s*(.*)",
    re.DOTALL,
)
# Same shields/text badges anywhere in body (merged comments: "**Issues:**\n- ![BADGE]...")
_SHIELD_ANYWHERE_RE = re.compile(
    r"!\[(BUG|SECURITY|WARNING|SUGGESTION|INFORMATIONAL)\]\(https://[^)]+\)\s*([^\n]*)",
    re.IGNORECASE,
)
_TEXT_BADGE_ANYWHERE_RE = re.compile(
    r"\*\*\[(BUG|SECURITY|WARNING|SUGGESTION|INFORMATIONAL)\]\*\*\s*([^\n]*)",
    re.IGNORECASE,
)
# GitHub may return HTML; badge alt text is often the severity label only.
_HTML_IMG_ALT_SEV_RE = re.compile(
    r'<img[^>]+alt=["\'](BUG|SECURITY|WARNING|SUGGESTION|INFORMATIONAL)["\']',
    re.IGNORECASE,
)

_SEVERITY_RANK = {"bug": 0, "security": 1, "warning": 2, "suggestion": 3, "informational": 4}


def _strip_merge_issues_header(text: str) -> str:
    """Remove leading **Issues:** line (merged multi-issue comments) so badges parse reliably."""
    t = text.strip().lstrip("\ufeff")
    t = re.sub(r"^\*\*Issues:\*\*\s*\n?", "", t, count=1, flags=re.IGNORECASE)
    t = re.sub(r"^-\s*\*\*Issues:\*\*\s*\n?", "", t, count=1, flags=re.IGNORECASE)
    return t.strip()


def _is_placeholder_issue_title(title: str) -> bool:
    """True if title is only a merge header / noise, not a real issue title."""
    if not title or not title.strip():
        return True
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", title)
    s = t.strip().strip("*").strip("_").strip()
    low = re.sub(r"\*+", "", s).lower().strip().rstrip(":").strip()
    return low in ("issues", "issue", "issues:", "issue:")


def _parse_issues_from_comment_body(text: str) -> List[tuple[str, str]]:
    """Extract (severity, title_fragment) from all shield/text badges in a comment body."""
    text = _strip_merge_issues_header(text)
    issues: List[tuple[str, str]] = []
    for m in _SHIELD_ANYWHERE_RE.finditer(text):
        sev = m.group(1).lower()
        title = (m.group(2) or "").strip()
        title = re.sub(r"^\s*[-*]\s+", "", title).strip("*").strip()
        if _is_placeholder_issue_title(title):
            title = ""
        issues.append((sev, title))
    if not issues:
        for m in _TEXT_BADGE_ANYWHERE_RE.finditer(text):
            sev = m.group(1).lower()
            title = (m.group(2) or "").strip().split("\n")[0].strip()
            title = re.sub(r"^\s*[-*]\s+", "", title).strip("*").strip()
            if _is_placeholder_issue_title(title):
                title = ""
            issues.append((sev, title))
    if not issues and "<img" in text.lower():
        # Fallback: HTML bodies (no markdown ![]() in body)
        for m in _HTML_IMG_ALT_SEV_RE.finditer(text):
            issues.append((m.group(1).lower(), ""))
    return issues

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


async def _call_llm(
    system: str,
    user_content: str,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """Call configured LLM provider; return assistant message content or empty string."""
    resolved_base = api_base or config.LLM_API_BASE or None
    resolved_model = model or config.LLM_MODEL
    resolved_key = api_key or config.LLM_API_KEY or None

    # When both a custom base and key are provided, call the OpenAI-compatible endpoint
    # directly — LiteLLM's ollama/ollama_chat providers don't forward Bearer auth.
    if resolved_key and resolved_base:
        import asyncio as _asyncio
        raw_model = resolved_model.split("/", 1)[-1] if "/" in resolved_model else resolved_model
        base = resolved_base.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        url = f"{base}/chat/completions"
        payload = {
            "model": raw_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {resolved_key}"}
        for attempt in range(5):
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.debug("429 rate-limit, retrying in %ss", wait)
                await _asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return (resp.json()["choices"][0]["message"]["content"] or "").strip()
        resp.raise_for_status()  # final attempt exhausted

    kwargs: Dict[str, Any] = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "api_base": resolved_base,
        "timeout": 120.0,
        "temperature": temperature,
    }
    response = await acompletion(**kwargs)
    return (response.choices[0].message.content or "").strip()


# Match line-number references so we can parse LLM output. Supports:
# "Line 42:", "line 42", "L42", "#42", "at line 42", "on line 42", "Line 42 -", "Line 42."
_LINE_REF_RE = re.compile(
    r"(?:^|\n)\s*(?:(?:Line|line|L)\s*|#\s*|(?:at|on)\s+line\s+)(\d+)\s*[:\.\-]?\s*",
    re.IGNORECASE,
)


def _strip_thinking_blocks(text: str) -> str:
    """Remove <thinking>…</thinking> and <reasoning>…</reasoning> blocks.

    Thinking/reasoning models (DeepSeek, Qwen3, o-series) emit these before
    their actual output. They confuse JSON parsers because the prose often
    contains '[', '{', ']', '}' characters.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


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
    ("bug", f"![BUG](https://img.shields.io/badge/BUG-AA0000?style={_BADGE_STYLE})"),
    ("security", f"![SECURITY](https://img.shields.io/badge/SECURITY-CC5500?style={_BADGE_STYLE})"),
    ("warning", f"![WARNING](https://img.shields.io/badge/WARNING-B8860B?style={_BADGE_STYLE})"),
    ("suggestion", f"![SUGGESTION](https://img.shields.io/badge/SUGGESTION-2E5A8A?style={_BADGE_STYLE})"),
    (
        "informational",
        f"![INFORMATIONAL](https://img.shields.io/badge/INFORMATIONAL-555555?style={_BADGE_STYLE})",
    ),
]
_SEV_BADGE_BY_KEY = {key: badge for key, badge in _SEV_META}

# Summary-only: shields path label-messageColor with dark labelColor (left) and lighter message (right).
_SEV_SUMMARY_SPEC: Dict[str, Dict[str, str]] = {
    "bug": {"label": "BUG", "message_color": "FF4444", "label_color": "AA0000"},
    "security": {"label": "SECURITY", "message_color": "FF8C00", "label_color": "CC5500"},
    "warning": {"label": "WARNING", "message_color": "FFD700", "label_color": "B8860B"},
    "suggestion": {"label": "SUGGESTION", "message_color": "4A90D9", "label_color": "2E5A8A"},
    "informational": {"label": "INFORMATIONAL", "message_color": "AAAAAA", "label_color": "555555"},
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
    _fix_raw = item.get("fix") or ""
    if isinstance(_fix_raw, dict):
        # Model returned {"before": "...", "after": "..."} or similar — render as before/after
        before = (_fix_raw.get("before") or "").strip()
        after = (_fix_raw.get("after") or _fix_raw.get("after_fix") or "").strip()
        fix = f"Before:\n{before}\n\nAfter:\n{after}" if before or after else str(_fix_raw)
    else:
        fix = str(_fix_raw).strip()
    parts = [f"{badge} {title}"]
    if body_text:
        parts.append("\n\n" + body_text)
    if fix:
        fix_clean = _strip_diff_markers_from_code_block(fix)
        parts.append("\n\n**Suggested fix:**\n\n```\n" + fix_clean + "\n```")
    return "".join(parts)


def _balanced_array_end(text: str, start: int) -> int:
    """Return the index just past the ']' that closes the '[' at `start`, or -1.

    String-aware: brackets inside quoted strings don't affect nesting depth.
    """
    depth = 0
    in_string: Optional[str] = None
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
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _extract_json_array(raw: str) -> Optional[List[Any]]:
    """Extract a JSON array from raw LLM output.

    Robust against reasoning models: first strips <think>/<thinking>/<reasoning>
    blocks, then tries each '[' as a candidate start and returns the first
    balanced slice that parses as a JSON *list*. This survives prose that
    contains stray brackets (e.g. "[L10]" line references), markdown fences, and
    leading commentary before the real array.

    Returns None when no list can be extracted. Callers should WARNING-log when
    the raw input was non-empty but this returns None — that signals a parse
    failure (output received but unusable), not a genuinely empty result.
    """
    text = _strip_thinking_blocks(raw)
    search_from = 0
    while True:
        start = text.find("[", search_from)
        if start < 0:
            return None
        end = _balanced_array_end(text, start)
        if end > 0:
            try:
                parsed = json.loads(text[start:end])
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return parsed
        search_from = start + 1


def _parse_review_file_response(raw: str, path: str) -> List[Dict[str, Any]]:
    """Parse LLM response: prefer JSON array; fall back to freeform line-ref regex."""
    if not raw or not raw.strip():
        return []
    text = raw.strip()

    # Try structured JSON first
    arr = _extract_json_array(text)
    if arr is not None and isinstance(arr, list):
        out: List[Dict[str, Any]] = []
        skipped = 0
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
            try:
                confidence = int(item.get("confidence", 7))
            except (TypeError, ValueError):
                confidence = 7
            logger.debug(
                "LLM finding: file=%s line=%s severity=%s title=%r confidence=%s",
                path, line_int, item.get("severity"), item.get("title"), confidence,
            )
            if confidence < 5:
                skipped += 1
                logger.debug(
                    "LLM finding SKIPPED (confidence=%s < 5): file=%s line=%s title=%r",
                    confidence, path, line_int, item.get("title"),
                )
                continue
            body = _format_structured_comment_body(item)
            if body.strip():
                out.append({
                    "line": line_int,
                    "body": body,
                    "post_inline": True,
                    "severity": (item.get("severity") or "suggestion").lower(),
                    "title": (item.get("title") or "").strip(),
                    "confidence": confidence,
                    "fix": item.get("fix") or None,
                })
        logger.debug(
            "LLM parse summary for %s: total=%d accepted=%d skipped_low_confidence=%d",
            path, len(arr), len(out), skipped,
        )
        if out:
            return out
        logger.debug("JSON array empty or invalid items for %s", path)

    # Fallback 1: tab-separated format "[L<n>]\t<severity>\t<title>\t<body>"
    # Some models (e.g. qwen3-coder-next) emit this instead of JSON.
    _TAB_LINE_RE = re.compile(r"^\[L(\d+)\]\t(.+)$", re.MULTILINE)
    tab_matches = _TAB_LINE_RE.findall(text)
    if tab_matches:
        out = []
        for line_str, rest in tab_matches:
            try:
                line_int = int(line_str)
            except ValueError:
                continue
            parts = rest.split("\t", 2)
            severity = parts[0].strip() if len(parts) > 0 else "warning"
            title    = parts[1].strip() if len(parts) > 1 else rest.strip()
            body_txt = parts[2].strip() if len(parts) > 2 else ""
            body = _format_structured_comment_body({
                "severity": severity,
                "title": title,
                "body": body_txt or title,
            })
            if body.strip():
                out.append({
                    "line": line_int,
                    "body": body,
                    "post_inline": True,
                    "severity": severity.lower(),
                    "title": title,
                    "confidence": 7,
                    "fix": None,
                })
                logger.debug("LLM tab-format finding: file=%s line=%s sev=%s title=%r", path, line_int, severity, title)
        if out:
            return out

    # Fallback 2: freeform "Line N:" parsing
    matches = list(_LINE_REF_RE.finditer(text))
    if not matches:
        # Non-empty model output that yielded no findings via ANY parser is a
        # parse failure, not a genuine empty result. Log loudly with a snippet
        # so silent zero-finding reviews are visible in the server log.
        logger.warning(
            "Review parse FAILURE for %s: received %d chars of model output but "
            "extracted no findings (no JSON array, tab-format, or line refs). "
            "Raw head: %r",
            path, len(raw), raw.strip()[:300],
        )
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
        # severity=None marks "no structured label"; consumers fall back to defaults
        out.append({
            "line": line_int,
            "body": body,
            "severity": None,
            "title": "",
            "confidence": 7,
            "fix": None,
        })
    return out


def _format_file_context(file_context: Dict[str, Any]) -> str:
    """Format surrounding file context (read-only) for the LLM: path, line ranges, and numbered lines."""
    path = file_context.get("path") or "?"
    content = (file_context.get("content") or "").strip()
    ranges = file_context.get("ranges") or []
    if not content:
        return ""
    lines = content.splitlines()
    # When the file is small enough, render it in full so the model can verify
    # cross-references (e.g. whether an import is used elsewhere) instead of
    # guessing from excerpts. Above the cap, fall back to the changed ranges only.
    if len(lines) <= config.SIFT_FULL_FILE_RENDER_MAX_LINES:
        out_lines: List[str] = [
            "Full file (read-only, for understanding the change):",
            f"File: {path}",
        ]
        for i, line in enumerate(lines):
            out_lines.append(f"  {i + 1:4d} | {line}")
        return "\n".join(out_lines).strip()
    if not ranges:
        return ""
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


def _format_repo_preferences(prefs_text: str) -> str:
    """Wrap repo-level feedback preferences for the user prompt (same section style as other blocks)."""
    return (prefs_text or "").strip()


def _format_caller_context(caller_infos: List[Any]) -> str:
    """Format PR-internal import/caller context for the LLM."""
    if not caller_infos:
        return ""
    lines = [
        "Other files in this PR import from changed modules below. "
        "Check that usages in this file remain compatible with their modifications:",
    ]
    for info in caller_infos:
        changed = getattr(info, "changed_path", None) or (
            info.get("changed_path") if isinstance(info, dict) else "?"
        )
        names = getattr(info, "function_names", None)
        if names is None and isinstance(info, dict):
            names = info.get("function_names") or ()
        names = tuple(names or ())
        if names:
            sym = ", ".join(f"`{n}`" for n in names)
            lines.append(f"- `{changed}` — modified symbols: {sym}")
        else:
            lines.append(f"- `{changed}` — module was modified (verify imports and call compatibility)")
    return "\n".join(lines)


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

    labeled_comments = (pr_context or {}).get("repo_feedback_labeled_comments")
    if labeled_comments:
        lc_block = _format_repo_preferences(labeled_comments)
        if lc_block:
            user_parts.append(lc_block)

    caller_context = (pr_context or {}).get("caller_context")
    if caller_context:
        cc_block = _format_caller_context(caller_context)
        if cc_block:
            user_parts.append(cc_block)

    semantic_ba = (pr_context or {}).get("semantic_before_after")
    if semantic_ba and str(semantic_ba).strip():
        user_parts.append(
            "Semantic before/after of changed functions in this file:\n"
            + str(semantic_ba).strip()
        )

    callee_sigs = (pr_context or {}).get("callee_signatures")
    if callee_sigs and str(callee_sigs).strip():
        user_parts.append(
            "Callee definitions from other files in this PR:\n"
            + str(callee_sigs).strip()
        )

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

    system_prompt = REVIEW_FILE_SYSTEM
    if (pr_context or {}).get("is_test"):
        system_prompt = REVIEW_FILE_SYSTEM + TEST_FILE_APPENDIX
    raw = await _call_llm(system_prompt, user_content)
    logger.debug("LLM raw output for %s:\n%s", path, raw)
    return _parse_review_file_response(raw, path)


def extract_comment_severity_and_title(body: str) -> tuple:
    """Extract (severity_key, title) from inline/summary comment body (badges anywhere).

    Merged comments (multiple shields after **Issues:**) use the highest severity as primary
    and join non-empty titles with " · ".
    """
    if not body or not body.strip():
        return ("suggestion", "")
    text = body.strip()
    issues = _parse_issues_from_comment_body(text)
    if issues:
        primary_sev = min(issues, key=lambda x: _SEVERITY_RANK.get(x[0], 9))[0]
        titles = [t for _, t in issues if t]
        composite = " · ".join(titles) if titles else ""
        if not composite:
            distinct = sorted({s for s, _ in issues}, key=lambda x: _SEVERITY_RANK.get(x, 9))
            composite = f"({len(issues)} issue(s): {', '.join(distinct)})" if len(distinct) > 1 else ""
        if not composite:
            composite = primary_sev
        if len(composite) > 256:
            composite = composite[:253] + "..."
        if _is_placeholder_issue_title(composite):
            composite = ""
        return (primary_sev, composite)

    m = _SUMMARY_SHIELD_RE.match(text)
    if m:
        return (m.group(1).lower(), (m.group(2) or "").strip().split("\n")[0].strip())
    m = _SUMMARY_SEVERITY_RE.match(text)
    if m:
        return (m.group(1).lower(), (m.group(2) or "").strip().split("\n")[0].strip())
    first = text.split("\n")[0].strip()
    if first.lower() in ("**issues:**", "## issues", "### issues") or _is_placeholder_issue_title(first):
        return ("suggestion", "")
    return ("suggestion", first)


def _build_structured_summary(comments: List[Dict[str, Any]]) -> str:
    """Build a structured summary with alert blocks and collapsible file details."""
    if not comments:
        return "Sifted through the code and found no issues."

    counts: Dict[str, int] = {
        "bug": 0,
        "security": 0,
        "warning": 0,
        "suggestion": 0,
        "informational": 0,
    }
    by_path: Dict[str, List[Dict[str, Any]]] = {}
    for c in comments:
        path = c.get("path", "?")
        line = c.get("line", "?")
        body = (c.get("body") or "").strip()
        if not body:
            continue
        sev, title = extract_comment_severity_and_title(body)
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
    informational_count = counts.get("informational", 0)

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
                f"> {' '.join(parts)}",
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

    if informational_count > 0:
        lines.extend(
            [
                "> [!NOTE]",
                f"> {_summary_count_badge_markdown('informational', informational_count)} informational note(s) available in the Files changed tab.",
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

    Cross-file insights are produced by the holistic pipeline pass (Phase 3) as inline
    findings; this function only builds the structured summary table.
    """
    return _build_structured_summary(comments)
