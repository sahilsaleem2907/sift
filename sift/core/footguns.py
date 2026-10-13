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

Seed set started tiny (the classes the sentry smoke test missed): Django QuerySet
negative slicing, multiprocessing spawn-Process isinstance, and datetime
JSON-serialization. Extended per the offline code-review-benchmark's other four golden
files (keycloak=Java, grafana=Go, discourse=Ruby, cal_dot_com=TypeScript/JavaScript) —
add entries as new misses are observed.
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
    # ---- Java (keycloak) ----
    Footgun(
        language="java",
        note=(
            "`Optional.get()` called without a preceding `isPresent()` (or `isEmpty()` guard) "
            "throws `NoSuchElementException` if the Optional is empty. Confirm the Optional is "
            "checked first, or use `.orElseThrow(...)`/`.orElse(...)` instead of a bare `.get()`."
        ),
        # bare no-arg `.get()` — distinct from Map.get(key) / List.get(i), which take an argument.
        pattern=re.compile(r"\.get\s*\(\s*\)"),
        context_any=("Optional",),
    ),
    Footgun(
        language="java",
        note=(
            "picocli `CommandLine.exit(...)` calls `System.exit(...)` directly — it terminates the "
            "whole JVM immediately, which is problematic inside a long-running process, a test, or "
            "anywhere that expects to return an exit code rather than kill the process."
        ),
        pattern=re.compile(r"\.exit\s*\("),
        context_any=("picocli", "CommandLine"),
    ),
    # ---- Ruby (discourse) ----
    Footgun(
        language="ruby",
        note=(
            "`Kernel#open(url)` (bare `open(...)`, not `File.open`) is SSRF-prone: if the argument "
            "can start with `|` it shells out, and if it is an http(s)/ftp URL it fetches it — so an "
            "attacker-influenced URL reaches arbitrary internal network fetches or command execution. "
            "Use `URI.open` only after validating the scheme/host, or a dedicated HTTP client."
        ),
        pattern=re.compile(r"(?<![.\w])open\s*\(\s*\w"),
        context_any=("url", "URI", "http", "Url"),
    ),
    Footgun(
        language="ruby",
        note=(
            "`X-Frame-Options: ALLOWALL` completely disables clickjacking protection for this "
            "response — any origin can frame the page. If framing must be allowed, scope it to a "
            "specific trusted origin (or CSP `frame-ancestors`) rather than ALLOWALL, and don't rely "
            "solely on `Referer` validation as a substitute (the header is attacker-controllable)."
        ),
        pattern=re.compile(r"ALLOWALL"),
        context_any=("X-Frame-Options", "X_FRAME_OPTIONS"),
    ),
    Footgun(
        language="ruby",
        note=(
            "An interpolated regex like `/@(#{domains})/` (or `Regexp.new(\"@(#{...})\")`) with no "
            "anchor matches a DOMAIN SUFFIX, not the full domain: `evil.example.com` matches an "
            "allow-list entry `example.com` because the pattern only requires `example.com` to appear "
            "somewhere after an `@`. Anchor it (`\\z` / `$`) so only an exact (sub)domain match passes."
        ),
        pattern=re.compile(r"/@\(#\{[^}]+\}\)(?!\\z)/"),
    ),
    Footgun(
        language="ruby",
        note=(
            "Appending to a STRING LITERAL with `<<` (e.g. `'.' << host`) mutates that literal "
            "object in place rather than building a new string. Under `frozen_string_literal: true` "
            "this raises `FrozenError` / `RuntimeError: can't modify frozen String`; even without it, "
            "a literal can be memoized/shared by the VM, so the mutation can leak across call sites. "
            "Use `'.' + host` or `\"#{'.'}#{host}\"` / string interpolation instead."
        ),
        pattern=re.compile(r"""(['"])[^'"]{0,3}\1\s*<<\s*\w"""),
    ),
    Footgun(
        language="ruby",
        note=(
            "Rails/ActiveModel serializer convention: a conditional-inclusion method must be named "
            "`include_<attr>?` (with the trailing `?`) or the serializer will not recognize it as the "
            "inclusion predicate for `<attr>`, silently always including (or never conditionally "
            "excluding) the attribute regardless of this method's return value."
        ),
        pattern=re.compile(r"def\s+include_\w+(?!\?)(?:\s*[\(\n;]|\s*$)"),
        context_any=("Serializer", "serializable", "attributes"),
    ),
    # ---- TypeScript / JavaScript (cal_dot_com; also covers discourse's client-side embed JS) ----
    Footgun(
        language="typescript",
        note=(
            "`Array#forEach` with an `async` callback does NOT await the returned promises — "
            "`forEach` ignores callback return values entirely, so each iteration's async work "
            "(and any error inside it) runs detached/fire-and-forget, un-sequenced, and unobservable "
            "to a surrounding try/catch. Use a `for...of` loop (sequential) or "
            "`await Promise.all(items.map(async ...))` (parallel, but still awaited)."
        ),
        pattern=re.compile(r"\.forEach\s*\(\s*async\b"),
    ),
    Footgun(
        language="typescript",
        note=(
            "`===` on two `dayjs(...)`/`moment(...)` (or `new Date(...)`) values compares OBJECT "
            "IDENTITY, not the represented time — two distinct instances for the same instant are "
            "never `===`, so this comparison is always `false`. Use `.isSame(...)` (dayjs/moment) or "
            "compare `.valueOf()`/`.getTime()` instead."
        ),
        pattern=re.compile(r"(?:dayjs|moment)\([^()]*\)(?:\.[a-zA-Z_]+\([^()]*\))*\s*===\s*(?:dayjs|moment)\("),
    ),
    Footgun(
        language="typescript",
        note=(
            "A membership/equality check via `.indexOf(...)`/`.includes(...)` against a "
            "security-sensitive value (backup code, blacklist/blocklist/denylist entry) without "
            "first normalizing case (`.toLowerCase()`/`.toUpperCase()`) is case-sensitive: it can "
            "both reject a legitimately-cased value and let a differently-cased value slip past a "
            "deny-list check."
        ),
        pattern=re.compile(r"\.(?:indexOf|includes)\s*\("),
        context_any=("backupCode", "backup_code", "BackupCode", "blacklist", "blocklist", "denylist"),
    ),
]

# JS-side footguns that are language-shared with TypeScript above but need to also fire on plain
# .js files (e.g. a server-rendered app's older embed/client script, not authored in TypeScript).
_FOOTGUNS.extend(
    Footgun(language="javascript", note=fg.note, pattern=fg.pattern, context_all=fg.context_all, context_any=fg.context_any)
    for fg in list(_FOOTGUNS)
    if fg.language == "typescript"
)

_EXT_TO_LANG = {
    ".py": "python", ".pyi": "python",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
    ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript",
}


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


# Top-level `def <name>` (ignores leading indentation so it also matches methods nested one
# level inside a class/module body, which is the common case).
_RUBY_DEF_RE = re.compile(r"^\s*def\s+(self\.)?([a-zA-Z_][a-zA-Z0-9_]*[?!=]?)", re.MULTILINE)

_DUPLICATE_DEF_NOTE = (
    "This method name is defined more than once in this file. In Ruby the LATER definition "
    "silently overrides the earlier one — there is no redefinition warning by default — so the "
    "first definition (and any callers relying on its distinct parameter list/behavior) becomes "
    "dead code, and only the last `def` for this name is ever reachable."
)


def _detect_duplicate_ruby_def(added: list[str], file_content: str) -> str:
    """Ruby-specific: a `def` added in this diff redefines a method already defined elsewhere
    in the file (same name, ignoring `self.` receiver), silently shadowing the earlier one.

    Not expressible as a declarative Footgun (those match "pattern present" / "context present",
    not "this identifier is defined more than once") — so this is a small dedicated pure check,
    following the module's design (sync, no I/O, matches on diff + file text only).
    """
    if not file_content:
        return ""
    added_names = {m.group(2) for ln in added for m in _RUBY_DEF_RE.finditer(ln)}
    if not added_names:
        return ""
    all_names: list[str] = [m.group(2) for m in _RUBY_DEF_RE.finditer(file_content)]
    for name in added_names:
        if all_names.count(name) > 1:
            return _DUPLICATE_DEF_NOTE
    return ""


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

    if lang == "ruby":
        dup_note = _detect_duplicate_ruby_def(added, file_content)
        if dup_note and dup_note not in notes:
            notes.append(dup_note)

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
