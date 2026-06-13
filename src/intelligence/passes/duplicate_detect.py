"""Intra-PR duplicate function detector — deterministic, LLM-free.

Catches copy-paste duplicate logic introduced within a single PR across different
files, in three stages:

  Stage 1 — Type-1 (exact):   group by existing FunctionChunk.content_hash
                               (whitespace-collapsed SHA256, already computed)
  Stage 2 — Type-2 (renamed): group by normalized_hash — alpha-renames all
                               local identifiers in the tree-sitter CST so
                               validate_amount(amount) == validate_amount(value)
  Stage 3 — Type-3 (near-miss): Jaccard token-shingle similarity (k=5) on
                               normalized token sequences; threshold ≥ 0.8

Each stage only processes functions not already flagged by a previous stage.
All findings are critic_exempt=True — guaranteed output, no LLM involved.
"""
from __future__ import annotations

import hashlib
import logging
import re
from itertools import combinations
from typing import Optional

from src.intelligence.ast.function_extract import FunctionChunk
from src.intelligence.ast.language_registry import detect_language_key, get_language_by_key
from src.intelligence.schema import Certainty, Finding, Impact

logger = logging.getLogger(__name__)

# Minimum size for a function to be considered for duplicate detection.
# Avoids false positives from trivially short functions (getters, stubs).
_MIN_LINES = 3
_MIN_TOKENS = 20

# Jaccard similarity threshold for Type-3 near-miss detection.
_JACCARD_THRESHOLD = 0.8

# k for token k-gram shingles.
_SHINGLE_K = 5

# Identifier node types per language — used for alpha-renaming in Type-2.
_IDENTIFIER_NODE_TYPES = frozenset({
    "identifier",         # Python, JS/TS, Go, Rust, Java, C, C++, Ruby, Kotlin
    "name",               # Python attribute targets
    "variable_name",      # PHP
    "simple_identifier",  # Kotlin/Swift
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _line_count(chunk: FunctionChunk) -> int:
    return chunk.end_line - chunk.start_line + 1


def _is_eligible(chunk: FunctionChunk) -> bool:
    """True if the function is large enough to be worth checking."""
    lines = _line_count(chunk)
    tokens = len(chunk.text.split())
    return lines >= _MIN_LINES or tokens >= _MIN_TOKENS


def _tokenize(text: str) -> list[str]:
    """Simple token split: split on whitespace + punctuation boundaries."""
    return re.findall(r"[A-Za-z_]\w*|\d+|[^\s\w]", text)


def normalized_hash(chunk: FunctionChunk) -> str:
    """Alpha-rename local identifiers in the tree-sitter CST, return SHA256.

    Falls back to content_hash if tree-sitter parsing is unavailable for the
    file's language — Type-1 detection still works in that case.
    """
    lang_key = detect_language_key(chunk.path, chunk.text)
    if not lang_key:
        return chunk.content_hash

    lang = get_language_by_key(lang_key)
    if lang is None:
        return chunk.content_hash

    try:
        from tree_sitter import Parser
        parser = Parser()
        parser.set_language(lang)
        tree = parser.parse(chunk.text.encode("utf-8"))
    except Exception:
        return chunk.content_hash

    # Walk the tree and collect identifier leaf text in order, then
    # alpha-rename: first seen identifier → v0, next new one → v1, etc.
    name_map: dict[str, str] = {}
    tokens: list[str] = []

    def _walk(node) -> None:
        if node.child_count == 0:
            text = node.text.decode("utf-8", errors="replace")
            if node.type in _IDENTIFIER_NODE_TYPES:
                if text not in name_map:
                    name_map[text] = f"v{len(name_map)}"
                tokens.append(name_map[text])
            else:
                tokens.append(text)
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    normalized = " ".join(tokens)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def token_shingles(tokens: list[str], k: int = _SHINGLE_K) -> frozenset:
    """Return frozenset of k-gram shingles (tuples) from a token list."""
    if len(tokens) < k:
        return frozenset()
    return frozenset(tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity between two shingle sets."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def _make_finding(
    primary: FunctionChunk,
    others: list[FunctionChunk],
    certainty: Certainty,
) -> Finding:
    """Build a critic_exempt Finding for a group of duplicate functions."""
    fn_name = primary.name or "<anonymous>"
    all_locations = [primary] + others
    first = all_locations[0]

    if len(all_locations) == 2:
        other = all_locations[1] if all_locations[1] is not primary else all_locations[0]
        title = f"Duplicate logic: {fn_name} also defined in {other.path}"
    else:
        title = f"Duplicate logic: {fn_name} defined in {len(all_locations)} locations"

    location_lines = "\n".join(
        f"  - `{c.path}` lines {c.start_line}–{c.end_line}"
        for c in all_locations
    )
    body = (
        f"The function `{fn_name}` appears to be duplicated across multiple files "
        f"in this PR:\n\n{location_lines}\n\n"
        f"Consider extracting the shared logic into a utility module to avoid "
        f"divergence bugs when one copy is updated but others are not."
    )

    return Finding(
        path=primary.path,
        line=primary.start_line,
        title=title[:80],
        body=body,
        impact=Impact.LOW,
        certainty=certainty,
        category="design",
        origin="duplicate_detect",
        fix=None,
        post_inline=True,
        critic_exempt=True,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def detect_duplicate_functions(
    mod_funcs_by_path: dict[str, list[FunctionChunk]],
) -> list[Finding]:
    """Detect duplicate functions across files in a single PR.

    Returns critic_exempt Findings — one per duplicate group, on the second
    (or later) occurrence. Empty list if no duplicates found or not enough
    eligible functions.
    """
    # Flatten all eligible functions with their file path
    all_chunks: list[FunctionChunk] = [
        chunk
        for chunks in mod_funcs_by_path.values()
        for chunk in chunks
        if _is_eligible(chunk)
    ]

    if len(all_chunks) < 2:
        return []

    findings: list[Finding] = []
    flagged_ids: set[int] = set()  # id(chunk) of already-flagged chunks

    # -----------------------------------------------------------------------
    # Stage 1 — Type-1: exact content hash match
    # -----------------------------------------------------------------------
    hash1_groups: dict[str, list[FunctionChunk]] = {}
    for chunk in all_chunks:
        hash1_groups.setdefault(chunk.content_hash, []).append(chunk)

    for h, group in hash1_groups.items():
        if len(group) < 2:
            continue
        # Only flag across different paths (same file duplicates are a different issue)
        paths = {c.path for c in group}
        if len(paths) < 2:
            continue
        # Emit finding on second+ occurrences, first is "original"
        original = group[0]
        duplicates = group[1:]
        for dup in duplicates:
            findings.append(_make_finding(dup, [original] + [d for d in duplicates if d is not dup], Certainty.CONFIRMED))
            flagged_ids.add(id(dup))
        flagged_ids.add(id(original))  # don't re-flag original in later stages
        logger.debug(
            "[duplicate_detect] Type-1 group: fn=%s paths=%s",
            group[0].name, [c.path for c in group],
        )

    # -----------------------------------------------------------------------
    # Stage 2 — Type-2: normalized (alpha-renamed) hash match
    # -----------------------------------------------------------------------
    unflagged = [c for c in all_chunks if id(c) not in flagged_ids]
    if len(unflagged) >= 2:
        hash2_groups: dict[str, list[FunctionChunk]] = {}
        for chunk in unflagged:
            nh = normalized_hash(chunk)
            hash2_groups.setdefault(nh, []).append(chunk)

        for nh, group in hash2_groups.items():
            if len(group) < 2:
                continue
            paths = {c.path for c in group}
            if len(paths) < 2:
                continue
            original = group[0]
            duplicates = group[1:]
            for dup in duplicates:
                findings.append(_make_finding(dup, [original] + [d for d in duplicates if d is not dup], Certainty.CONFIRMED))
                flagged_ids.add(id(dup))
            flagged_ids.add(id(original))
            logger.debug(
                "[duplicate_detect] Type-2 group: fn=%s paths=%s",
                group[0].name, [c.path for c in group],
            )

    # -----------------------------------------------------------------------
    # Stage 3 — Type-3: Jaccard token-shingle near-miss
    # -----------------------------------------------------------------------
    unflagged = [c for c in all_chunks if id(c) not in flagged_ids]
    if len(unflagged) >= 2:
        # Precompute shingle sets for each unflagged chunk
        shingle_cache: dict[int, frozenset] = {
            id(c): token_shingles(_tokenize(c.text))
            for c in unflagged
        }

        for a, b in combinations(unflagged, 2):
            if a.path == b.path:
                continue  # skip same-file pairs
            if id(a) in flagged_ids or id(b) in flagged_ids:
                continue
            sim = jaccard(shingle_cache[id(a)], shingle_cache[id(b)])
            if sim >= _JACCARD_THRESHOLD:
                # Emit finding on the second chunk (b, later in iteration)
                findings.append(_make_finding(b, [a], Certainty.LIKELY))
                flagged_ids.add(id(a))
                flagged_ids.add(id(b))
                logger.debug(
                    "[duplicate_detect] Type-3 pair: fn_a=%s fn_b=%s sim=%.2f",
                    a.name, b.name, sim,
                )

    if findings:
        logger.info("[duplicate_detect] %d duplicate finding(s) emitted", len(findings))

    return findings
