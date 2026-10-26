# SPDX-License-Identifier: AGPL-3.0-only
"""Context-window budgeting: token estimates, block trimming, diff chunking, overflow detection.

Every prompt that carries a file diff is sized against the model's context window here, so a
large file degrades (drops optional context, then splits into chunks) instead of failing with a
provider "prompt is too long" error and ending up unreviewed.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import litellm

from sift import config

logger = logging.getLogger(__name__)

# Output tokens reserved inside the window: reasoning calls request max_tokens=16000
# (see llm_client._call_llm); plain calls rarely exceed a few thousand.
OUTPUT_RESERVE_REASONING = 16000
OUTPUT_RESERVE_DEFAULT = 4096
_MIN_INPUT_BUDGET = 2048

# Applied to a retry budget when the provider rejects a prompt without saying by how much.
_UNKNOWN_OVERFLOW_SHRINK = 0.7

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_OVERFLOW_PATTERNS = (
    # Ollama / vLLM: "The prompt is too long: 147021, model maximum context length: 131072"
    re.compile(r"prompt is too long:\s*(?P<actual>\d+).*?maximum context length:\s*(?P<max>\d+)", re.I | re.S),
    # OpenAI: "maximum context length is 131072 tokens. However, your messages resulted in 147021 tokens"
    re.compile(r"maximum context length is\s*(?P<max>\d+).*?resulted in\s*(?P<actual>\d+)", re.I | re.S),
    # Anthropic: "prompt is too long: 210000 tokens > 200000 maximum"
    re.compile(r"prompt is too long:\s*(?P<actual>\d+)\s*tokens\s*>\s*(?P<max>\d+)", re.I | re.S),
)


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Estimated token count. LiteLLM falls back to cl100k for models it doesn't know."""
    if not text:
        return 0
    try:
        return int(litellm.token_counter(model=model or config.LLM_MODEL, text=text))
    except Exception as exc:
        logger.debug("[budget] token_counter failed (%s); using chars/3", exc)
        return len(text) // 3 + 1


def input_budget(context_window: int, reasoning: bool) -> int:
    """Tokens a prompt (system + user) may use for a model with this context window."""
    reserve = OUTPUT_RESERVE_REASONING if reasoning else OUTPUT_RESERVE_DEFAULT
    usable = int(context_window * config.SIFT_CONTEXT_SAFETY_FACTOR) - reserve
    return max(_MIN_INPUT_BUDGET, usable)


def overflow_shrink_factor(exc: BaseException) -> Optional[float]:
    """For a provider context-length rejection, the factor to scale the budget by; else None."""
    msg = str(exc)
    for pattern in _OVERFLOW_PATTERNS:
        m = pattern.search(msg)
        if m:
            actual, maximum = int(m.group("actual")), int(m.group("max"))
            if actual > 0 and maximum > 0:
                return min(0.9, maximum / actual * 0.9)
    if isinstance(exc, litellm.ContextWindowExceededError):
        return _UNKNOWN_OVERFLOW_SHRINK
    return None


# ---- prompt blocks ----


@dataclass
class Block:
    """One section of a prompt.

    ``steps`` are the reductions this block allows, as (rank, label, replacement) — lower ranks
    are applied first across all blocks; an empty replacement drops the block. A block with no
    steps is never reduced.
    """

    name: str
    text: str
    steps: list[tuple[int, str, str]] = field(default_factory=list)


def fit_blocks(
    blocks: list[Block],
    budget: int,
    overhead_tokens: int = 0,
    model: Optional[str] = None,
) -> tuple[list[Block], list[str]]:
    """Apply block reductions in rank order until the blocks fit ``budget - overhead_tokens``.

    Returns the non-empty blocks (in their original order) and the labels of the reductions
    applied. Stops when nothing is left to reduce, so the result can still be over budget.
    """
    texts = [b.text for b in blocks]
    sizes = [count_tokens(t, model) for t in texts]
    limit = budget - overhead_tokens
    applied: list[str] = []
    pending = sorted(
        ((rank, i, label, repl) for i, b in enumerate(blocks) for rank, label, repl in b.steps),
        key=lambda s: (s[0], s[1]),
    )
    for _rank, i, label, repl in pending:
        if sum(sizes) <= limit:
            break
        if not texts[i]:
            continue
        if repl and len(repl) >= len(texts[i]):
            continue
        texts[i] = repl
        sizes[i] = count_tokens(repl, model)
        applied.append(label)
    kept = [Block(b.name, t) for b, t in zip(blocks, texts) if t]
    return kept, applied


# ---- tool-loop message history ----

# Per tool result; a single huge result (a big file read, a broad grep) otherwise eats the window.
TOOL_RESULT_MAX_TOKENS = 8000


def cap_tool_result(text: str, max_tokens: int = TOOL_RESULT_MAX_TOKENS) -> str:
    """Truncate a tool result to roughly ``max_tokens``, noting the truncation."""
    if not text or len(text) <= max_tokens * 3:
        return text
    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return text
    keep = int(len(text) * max_tokens / tokens)
    return text[:keep] + f"\n… (tool result truncated: {tokens - max_tokens} more tokens omitted)"


def messages_tokens(messages: list[dict], model: Optional[str] = None) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):  # cache_control content blocks
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        total += count_tokens(content or "", model)
        for tc in m.get("tool_calls") or []:
            total += count_tokens((tc.get("function") or {}).get("arguments") or "", model)
    return total


def drop_tool_turns(messages: list[dict], budget: int, keep: int = 2) -> int:
    """Pop trailing tool turns (tool results + the assistant call that requested them).

    Used after a context-length rejection mid-loop: pops at least one turn, then keeps popping
    while the history is over ``budget``. The first ``keep`` messages (system + initial user
    turn) are never removed. Returns the number of turns popped.
    """
    popped = 0
    while len(messages) > keep and (popped == 0 or messages_tokens(messages) > budget):
        while len(messages) > keep and messages[-1].get("role") == "tool":
            messages.pop()
        if len(messages) > keep and messages[-1].get("role") == "assistant":
            messages.pop()
        popped += 1
    return popped


# ---- diff chunking ----


@dataclass
class DiffChunk:
    text: str
    new_start: int
    new_end: int


@dataclass
class _Hunk:
    old_start: int
    new_start: int
    section: str
    lines: list[str]


def _parse_diff(file_diff: str) -> tuple[list[str], list[_Hunk]]:
    header: list[str] = []
    hunks: list[_Hunk] = []
    for line in file_diff.splitlines():
        m = _HUNK_RE.match(line)
        if m:
            hunks.append(_Hunk(int(m.group(1)), int(m.group(3)), m.group(5), []))
        elif hunks:
            hunks[-1].lines.append(line)
        else:
            header.append(line)
    return header, hunks


def is_new_file_diff(file_diff: str) -> bool:
    """True when the diff only adds a file (no old side), so it already carries every line."""
    header, hunks = _parse_diff(file_diff or "")
    if not hunks:
        return False
    if any(h.startswith("--- /dev/null") or h.startswith("new file mode") for h in header):
        return True
    return all(
        not ln or ln[0] in "+\\"
        for h in hunks
        for ln in h.lines
    ) and all(h.old_start == 0 for h in hunks)


def _render_piece(old_no: int, new_no: int, section: str, lines: list[str]) -> str:
    old_count = sum(1 for ln in lines if ln[:1] in (" ", "-") or ln == "")
    new_count = sum(1 for ln in lines if ln[:1] in (" ", "+") or ln == "")
    old_start = old_no if old_count else max(0, old_no - 1)
    new_start = new_no if new_count else max(0, new_no - 1)
    head = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{section}"
    return "\n".join([head, *lines])


def _split_hunk(
    hunk: _Hunk, char_limit: int, func_ends: set[int]
) -> list[tuple[str, int, int]]:
    """Split one hunk into pieces of at most ``char_limit`` chars, each with its own header.

    Cuts prefer the end of a modified function, then a blank line, then any line; a cut is
    never placed right before a "\\ No newline" marker. Returns (text, new_start, new_end).
    """
    lines = hunk.lines
    n = len(lines)
    # Line numbers in effect *before* each body line, plus the new-file number each line has.
    old_before, new_before, new_at = [], [], []
    old_no, new_no = hunk.old_start, hunk.new_start
    for ln in lines:
        old_before.append(old_no)
        new_before.append(new_no)
        prefix = ln[:1]
        if prefix in (" ", ""):
            new_at.append(new_no)
            old_no += 1
            new_no += 1
        elif prefix == "+":
            new_at.append(new_no)
            new_no += 1
        elif prefix == "-":
            new_at.append(None)
            old_no += 1
        else:
            new_at.append(None)

    def last_new(start: int, end: int) -> Optional[int]:
        for k in range(end - 1, start - 1, -1):
            if new_at[k] is not None:
                return new_at[k]
        return None

    pieces: list[tuple[str, int, int]] = []
    start = 0
    while start < n:
        size = 0
        end = start
        while end < n and (end == start or size + len(lines[end]) + 1 <= char_limit):
            size += len(lines[end]) + 1
            end += 1
        if end < n:
            floor = start + max(1, (end - start) // 2)
            candidates = [c for c in range(end, floor - 1, -1) if not lines[c].startswith("\\")]
            cut = next((c for c in candidates if last_new(start, c) in func_ends), None)
            if cut is None:
                cut = next((c for c in candidates if not lines[c - 1][1:].strip()), None)
            if cut is None:
                cut = candidates[0] if candidates else end
            end = cut
        piece = lines[start:end]
        new_nums = [x for x in new_at[start:end] if x is not None]
        first = new_nums[0] if new_nums else new_before[start]
        last = new_nums[-1] if new_nums else new_before[start]
        pieces.append((_render_piece(old_before[start], new_before[start], hunk.section, piece), first, last))
        start = end
    return pieces


def split_diff(
    file_diff: str,
    budget_tokens: int,
    model: Optional[str] = None,
    func_ranges: Optional[Iterable[tuple[int, int]]] = None,
) -> list[DiffChunk]:
    """Split a file diff into chunks that each fit ``budget_tokens``.

    Whole hunks are packed greedily; a hunk larger than the budget (e.g. a new file, which is a
    single hunk) is split inside with recomputed headers, so ``[L<n>]`` annotation and line
    numbers stay correct. Every chunk keeps the file header. Chunks do not overlap.
    """
    header, hunks = _parse_diff(file_diff or "")
    if not hunks:
        return [DiffChunk(file_diff, 0, 0)]
    total_tokens = count_tokens(file_diff, model)
    chars_per_token = max(1.0, len(file_diff) / max(1, total_tokens))
    header_text = "\n".join(header)
    char_limit = max(200, int(budget_tokens * chars_per_token) - len(header_text) - 1)
    func_ends = {end for _start, end in (func_ranges or [])}

    # A hunk that fits comes back as one piece; an oversized one is split inside.
    items = [piece for h in hunks for piece in _split_hunk(h, char_limit, func_ends)]

    chunks: list[DiffChunk] = []
    cur: list[tuple[str, int, int]] = []
    cur_size = 0
    for item in items:
        if cur and cur_size + len(item[0]) + 1 > char_limit:
            chunks.append(_make_chunk(header_text, cur))
            cur, cur_size = [], 0
        cur.append(item)
        cur_size += len(item[0]) + 1
    if cur:
        chunks.append(_make_chunk(header_text, cur))
    return chunks


def _make_chunk(header_text: str, items: list[tuple[str, int, int]]) -> DiffChunk:
    body = "\n".join(text for text, _s, _e in items)
    text = f"{header_text}\n{body}" if header_text else body
    return DiffChunk(text, min(s for _t, s, _e in items), max(e for _t, _s, e in items))


__all__ = [
    "Block",
    "DiffChunk",
    "cap_tool_result",
    "count_tokens",
    "drop_tool_turns",
    "fit_blocks",
    "input_budget",
    "is_new_file_diff",
    "overflow_shrink_factor",
    "split_diff",
]
