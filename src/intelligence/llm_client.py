"""Local Ollama integration for code review."""
import logging
from typing import Any, Dict, Optional

import httpx

from src import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a concise code reviewer. Given a git diff, provide a short professional review:
- A few bullet points on correctness, style, and possible improvements.
- Be brief and actionable. Do not repeat the diff."""


async def review(diff: str, pr_context: Optional[Dict[str, Any]] = None) -> str:
    """Call Ollama to generate a code review for the given diff.

    pr_context may contain "title" and "body" for the PR description.
    Returns the model's review text.
    """
    base_url = config.OLLAMA_BASE_URL
    model = config.OLLAMA_MODEL

    user_content = diff
    if pr_context:
        title = pr_context.get("title") or ""
        body = pr_context.get("body") or ""
        if title or body:
            user_content = f"PR title: {title}\n\nPR description:\n{body}\n\n---\n\nDiff:\n{diff}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
        return "Review could not be generated."

    return (msg["content"] or "").strip()
