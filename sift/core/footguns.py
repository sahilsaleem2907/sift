"""Curated external-API 'footgun' notes for LLM grounding.

The code-intel tools are repo-only, so the reviewer is blind to stdlib/framework
runtime gotchas that are neither type-checkable (pyright can't see them) nor always
recalled from memory. This module is a small, curated registry of such gotchas: when
a file's changed lines match a known-dangerous pattern in the relevant context, we
inject an AUTHORITATIVE note into the reviewer/critic context.

Design mirrors version_detect.py: pure and sync (matches on the diff text + file
content, no I/O), so it is trivially unit-testable. Notes are advisory only — they
do NOT auto-create findings; the model still decides, and a matching note may later
corroborate a model finding (see the pipeline's corroboration step).

Seed set is intentionally tiny (the classes the sentry smoke test missed):
Django QuerySet negative slicing, multiprocessing spawn-Process isinstance, and
datetime JSON-serialization. Add entries as new misses are observed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Footgun:
    language: str
    note: str
    # Regex that must match at least one ADDED ('+') line in the diff.
    pattern: re.Pattern
    # Every substring here must appear somewhere in the file (context gate).
    context_all: tuple[str, ...] = ()
    # At least one of these substrings must appear in the file (context gate).
    context_any: tuple[str, ...] = ()


_FOOTGUNS: list[Footgun] = [
    Footgun(
        language="python",
        note=(
            "Django QuerySet negative indexing/slicing: if a subscript here is applied to a "
            "Django QuerySet (not a list/tuple), negative indices or a negative slice start "
            "raise `AssertionError: Negative indexing is not supported.` — QuerySets only "
            "support non-negative slicing. A paginator offset that can go negative will crash."
        ),
        # a subscript with a NON-empty start: `qs[start:...]`, `qs[-5:]`, `x[offset:limit]`
        # (deliberately does not match `x[:5]` or `x[:-1]`, whose start is empty).
        pattern=re.compile(r"\w\[\s*[^\s:\]][^\]]*:"),
        context_any=("QuerySet", ".objects", "order_by", ".filter(", "paginat", "Paginat"),
    ),
    Footgun(
        language="python",
        note=(
            "multiprocessing spawn context: `multiprocessing.get_context('spawn').Process` "
            "returns a `SpawnProcess`, which on POSIX is NOT a subclass of "
            "`multiprocessing.Process`. `isinstance(proc, multiprocessing.Process)` is "
            "therefore always False for spawn-created processes, so such a check silently "
            "never matches."
        ),
        pattern=re.compile(r"isinstance\s*\([^,]+,\s*(?:mp\.|multiprocessing\.)?Process\b"),
        context_all=("multiprocessing", "spawn"),
    ),
    Footgun(
        language="python",
        note=(
            "datetime is not JSON-serializable: `json.dumps(...)` on a value containing a "
            "`datetime` raises `TypeError: Object of type datetime is not JSON serializable` "
            "unless a custom `default`/encoder is supplied. This also breaks Celery "
            "`apply_async` kwargs when the JSON serializer is used."
        ),
        pattern=re.compile(r"json\.dumps\s*\("),
        context_any=("datetime", "timezone.now", "DateTime", "isoformat"),
    ),
]

_EXT_TO_LANG = {".py": "python", ".pyi": "python"}


def _language_for_path(path: str) -> str:
    p = path.replace("\\", "/").lower()
    for ext, lang in _EXT_TO_LANG.items():
        if p.endswith(ext):
            return lang
    return ""


def _added_lines(file_diff: str) -> list[str]:
    return [
        ln[1:]
        for ln in (file_diff or "").splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]


def detect_footguns(path: str, file_diff: str, file_content: str = "") -> list[str]:
    """Return authoritative notes for footgun patterns matched in this file's diff.

    A footgun fires when: the file's language matches, its context gates are satisfied
    (searched over file_content, falling back to the added lines), and its pattern
    matches at least one added line. Order-preserving, de-duplicated.
    """
    lang = _language_for_path(path)
    if not lang:
        return []
    added = _added_lines(file_diff)
    if not added:
        return []
    haystack = file_content or "\n".join(added)

    notes: list[str] = []
    for fg in _FOOTGUNS:
        if fg.language != lang:
            continue
        if fg.context_all and not all(c in haystack for c in fg.context_all):
            continue
        if fg.context_any and not any(c in haystack for c in fg.context_any):
            continue
        if any(fg.pattern.search(ln) for ln in added):
            if fg.note not in notes:
                notes.append(fg.note)
    return notes


def format_footgun_notes(notes: list[str]) -> str:
    """Render notes as a prompt block, or '' when there are none."""
    if not notes:
        return ""
    body = "\n".join(f"- {n}" for n in notes)
    return (
        "Known runtime footguns for external APIs used in the changed lines "
        "(authoritative — check the code against each):\n" + body
    )
