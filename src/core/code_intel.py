"""Repo-wide code-intelligence backends for the review fact-tools.

These power the LLM-callable tools (read_file, search_repo, find_definition,
find_callers, get_signature, get_mro). They operate on the PR's git checkout so
they can reach UNCHANGED code — the definitions and types the diff depends on.

Search is backed by `git grep` (the checkout is a git repo, so it is always
available and fast); structure (signatures, MRO, abstract methods) is backed by the
existing tree-sitter parser. Every function is defensive: any failure returns a
short "unavailable" string rather than raising, so a tool call never breaks a review.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from src.intelligence.ast.parser import parse_source

logger = logging.getLogger(__name__)

_MAX_MATCHES = 40
_MAX_READ_LINES = 160
_GREP_TIMEOUT = 20

# Appended when a symbol resolves to nothing in the checkout. The repo tools are
# repo-only (git grep + tree-sitter), so a stdlib/framework/third-party symbol is
# structurally invisible to them — a bare "not found" makes the model treat a real
# external-API bug as unconfirmable and stay silent. This tells it to fall back to
# its own knowledge for out-of-repo symbols.
_EXTERNAL_HINT = (
    "This may be an external symbol (standard library, framework, or third-party) "
    "that lives outside this repository, so the repo tools cannot inspect it. If so, "
    "rely on your own knowledge of that library at the stated target runtime version "
    "to decide — do not treat 'not found here' as proof it is safe."
)


def _safe_path(repo_root: str, path: str) -> Optional[Path]:
    """Resolve `path` under repo_root, refusing escapes outside the checkout."""
    try:
        root = Path(repo_root).resolve()
        target = (root / path).resolve()
        target.relative_to(root)  # raises if outside root
        return target
    except (ValueError, OSError):
        return None


def _git_grep(repo_root: str, args: List[str]) -> List[str]:
    """Run `git grep` in repo_root; return matching lines (capped). [] on failure."""
    try:
        proc = subprocess.run(
            ["git", "grep", "--no-color", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=_GREP_TIMEOUT,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug("[code_intel] git grep failed: %s", e)
        return []
    # git grep exits 1 when there are no matches — that is not an error.
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    return lines[:_MAX_MATCHES]


def read_file(repo_root: str, path: str, start: Optional[int] = None, end: Optional[int] = None) -> str:
    """Return up to _MAX_READ_LINES of a repo file (1-based, inclusive range)."""
    target = _safe_path(repo_root, path)
    if target is None or not target.is_file():
        return f"[not found: {path}]"
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return f"[unreadable: {path} ({e})]"
    if not lines:
        return "[file is empty]"
    s = max(1, start or 1)
    e = min(len(lines), end or (s + _MAX_READ_LINES - 1))
    if e - s + 1 > _MAX_READ_LINES:
        e = s + _MAX_READ_LINES - 1
    body = "\n".join(f"{i:5d} | {lines[i - 1]}" for i in range(s, e + 1))
    tail = "" if e >= len(lines) else f"\n... ({len(lines) - e} more lines)"
    return body + tail


def search_repo(repo_root: str, pattern: str) -> str:
    """Regex-search the repo (git grep). Returns `path:line: text` matches."""
    if not pattern.strip():
        return "[empty pattern]"
    matches = _git_grep(repo_root, ["-nE", pattern])
    if not matches:
        return f"[no matches for /{pattern}/]"
    return "\n".join(matches)


def find_definition(repo_root: str, symbol: str) -> str:
    """Find where `symbol` is defined (def/class/assignment) across the repo."""
    sym = symbol.strip()
    if not sym.isidentifier():
        return "[symbol must be a bare identifier]"
    # def/class in Python, plus common decl forms in other langs (func/class/type).
    # NOTE: git grep -E is POSIX ERE — no \s or \b; use [[:space:]] and explicit
    # boundary groups instead.
    b = r"([^[:alnum:]_]|$)"
    pattern = rf"(^|[[:space:]])(def|class|func|function|type|interface|struct)[[:space:]]+{sym}{b}"
    matches = _git_grep(repo_root, ["-nE", pattern])
    if not matches:
        return f"[no definition found for '{sym}' in this repo. {_EXTERNAL_HINT}]"
    return "\n".join(matches)


def find_callers(repo_root: str, symbol: str) -> str:
    """Find call/usage sites of `symbol` across the repo."""
    sym = symbol.strip()
    if not sym.isidentifier():
        return "[symbol must be a bare identifier]"
    matches = _git_grep(repo_root, ["-nE", rf"(^|[^[:alnum:]_]){sym}[[:space:]]*\("])
    if not matches:
        return f"[no callers found for '{sym}']"
    return "\n".join(matches)


# ---------- tree-sitter structure helpers ----------

def _iter(node) -> "Iterator[dict]":
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.get("children", []) or [])


def _child_text(node, types: Tuple[str, ...]) -> Optional[str]:
    for c in node.get("children", []) or []:
        if c.get("type") in types:
            return c.get("text")
    return None


def _read_source(repo_root: str, path: str) -> Optional[str]:
    target = _safe_path(repo_root, path)
    if target is None or not target.is_file():
        return None
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def get_signature(repo_root: str, symbol: str) -> str:
    """Return the definition line(s) of `symbol` (its signature) if resolvable."""
    defs = find_definition(repo_root, symbol)
    if defs.startswith("["):
        return defs
    # find_definition already returns `path:line: <def ...>` — that IS the signature.
    return defs


def _class_methods(class_node) -> Tuple[List[str], List[str]]:
    """Return (all_method_names, abstract_method_names) for a class AST node."""
    body = None
    for c in class_node.get("children", []) or []:
        if c.get("type") == "block":
            body = c
            break
    if body is None:
        return [], []
    methods: List[str] = []
    abstract: List[str] = []
    for stmt in body.get("children", []) or []:
        fn = None
        decorated_abstract = False
        if stmt.get("type") == "function_definition":
            fn = stmt
        elif stmt.get("type") == "decorated_definition":
            for c in stmt.get("children", []) or []:
                if c.get("type") == "decorator" and "abstractmethod" in (c.get("text") or ""):
                    decorated_abstract = True
                if c.get("type") == "function_definition":
                    fn = c
        if fn is None:
            continue
        name = _child_text(fn, ("identifier",))
        if not name:
            continue
        methods.append(name)
        # abstract if @abstractmethod OR body is just `...`/`pass`/docstring
        if decorated_abstract:
            abstract.append(name)
    return methods, abstract


def _find_class_node(root, class_name: str):
    for n in _iter(root):
        if n.get("type") == "class_definition":
            name = _child_text(n, ("identifier",))
            if name == class_name:
                return n
    return None


def _class_bases(class_node) -> List[str]:
    bases: List[str] = []
    for c in class_node.get("children", []) or []:
        if c.get("type") == "argument_list":
            for a in c.get("children", []) or []:
                if a.get("type") in ("identifier", "attribute"):
                    txt = a.get("text")
                    if txt:
                        bases.append(txt.split(".")[-1])
    return bases


def get_mro(repo_root: str, path: str, class_name: str) -> str:
    """Report base classes, each base's abstract methods, and which are unimplemented.

    Resolves each base repo-wide (its class may live in an unchanged file), so this
    grounds abstract-method-completeness (cat 2) and isinstance-hierarchy (cat 6).
    """
    src = _read_source(repo_root, path)
    if src is None:
        return f"[not found: {path}]"
    root = parse_source(path, src, max_text_len=400)
    if root is None:
        return f"[could not parse {path}]"
    cls = _find_class_node(root, class_name)
    if cls is None:
        return f"[class '{class_name}' not found in {path}]"

    own_methods, _ = _class_methods(cls)
    bases = _class_bases(cls)
    lines = [f"class {class_name}({', '.join(bases) or 'object'})"]
    lines.append(f"  own methods: {', '.join(own_methods) or '(none — body is empty/pass)'}")

    unimplemented: List[str] = []
    for base in bases:
        base_hits = _git_grep(repo_root, ["-lE", rf"(^|[[:space:]])class[[:space:]]+{base}([^[:alnum:]_]|$)"])
        if not base_hits:
            lines.append(
                f"  base {base}: [definition not found in repo — likely external "
                f"(stdlib/framework/third-party); rely on your knowledge of {base} "
                "and its abstract methods / type hierarchy at the target runtime]"
            )
            continue
        base_file = base_hits[0]
        base_src = _read_source(repo_root, base_file)
        if base_src is None:
            continue
        base_root = parse_source(base_file, base_src, max_text_len=400)
        if base_root is None:
            continue
        base_cls = _find_class_node(base_root, base)
        if base_cls is None:
            continue
        _, base_abstract = _class_methods(base_cls)
        missing = [m for m in base_abstract if m not in own_methods]
        lines.append(
            f"  base {base} ({base_file}): abstract={base_abstract or '(none)'} "
            f"unimplemented_by_{class_name}={missing or '(none)'}"
        )
        unimplemented.extend(missing)

    if unimplemented:
        lines.append(
            f"  ⚠ {class_name} does not implement abstract method(s): "
            f"{', '.join(sorted(set(unimplemented)))} — instantiation raises TypeError."
        )
    return "\n".join(lines)
