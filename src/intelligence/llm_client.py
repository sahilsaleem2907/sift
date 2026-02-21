"""Local Ollama integration for code review."""
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from src import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a concise code reviewer. Given a git diff, provide a short professional review:
- A few bullet points on correctness, style, and possible improvements.
- Be brief and actionable. Do not repeat the diff."""

REVIEW_FILE_SYSTEM = """You are a code reviewer. Given a single file's diff, write your review in plain text or markdown.

For each issue: mention the line number (the line number in the new/right side of the diff), describe the issue, and when helpful include a suggested fix as a code block so the developer can copy and paste it. Write naturally; no specific format required."""

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


# Match line-number references at start of line or after newline: "Line 42:", "line 42", "L42", "#42", "at line 42"
_LINE_REF_RE = re.compile(
    r"(?:^|\n)\s*(?:(?:Line|line|L)\s*|#\s*|at\s+line\s+)(\d+)\s*[:\.\-]?\s*",
    re.IGNORECASE,
)


def _normalize_comment_body(body: str) -> str:
    """Ensure body has **Suggested fix:** before code blocks and consistent formatting."""
    if not body or not body.strip():
        return body
    text = body.strip()
    # If there's a fenced code block but no "Suggested fix" / "Optimal solution" before it, add one
    if re.search(r"```", text) and not re.search(
        r"\*\*(?:Suggested fix|Optimal solution)\*\*", text, re.IGNORECASE
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


async def review_file(
    diff_chunk: str,
    path: str,
    pr_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Review a single file's diff; return list of {line, body} for inline comments.

    path is for logging; we do not rely on LLM for path. Line is the line number in the new file (right side).
    """
    user_content = f"File: {path}\n\nDiff:\n{diff_chunk}"
    if pr_context:
        title = pr_context.get("title") or ""
        body = pr_context.get("body") or ""
        if title or body:
            user_content = f"PR title: {title}\n\nPR description: {body}\n\n---\n\n{user_content}"
    raw = await _call_ollama(REVIEW_FILE_SYSTEM, user_content)
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
