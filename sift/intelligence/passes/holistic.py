# SPDX-License-Identifier: AGPL-3.0-only
"""Pass 3: whole-PR holistic review for cross-file and design-level issues."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from sift import config
from sift.intelligence.capability import ModelCapability
from sift.intelligence.effort import EffortPlan
from sift.intelligence.llm_client import _call_llm, _extract_json_array
from sift.intelligence.prompts import HOLISTIC_SYSTEM
from sift.intelligence.schema import (
    CATEGORIES,
    Certainty,
    Finding,
    Impact,
    derive_severity,
)

logger = logging.getLogger(__name__)

_MAX_CHANGED_FUNCTIONS = 30
_MAX_PER_FILE_DIGEST = 20
# Contract-drift into files NOT touched by this PR (e.g. a Lark/Office365 implementer of an
# interface whose method signature changed elsewhere) requires resolving real callers/co-definers
# via find_callers — capped to bound git-grep calls and prompt size.
_MAX_EXTERNAL_CALLER_SYMBOLS = 20
_MAX_EXTERNAL_CALLERS_PER_SYMBOL = 3
_MAX_EXTERNAL_CALLERS_TOTAL = 15


@dataclass
class PRDigest:
    title: str
    body: str
    changed_functions: list[dict]
    import_edges: list[dict]
    per_file_findings: list[dict]
    # raw added lines per path; shown in digest when function extraction is unavailable
    diff_excerpts: list[dict] = None  # list of {"path": str, "added_lines": str}
    # Real (find_callers-verified) usages of a changed symbol in files this PR did NOT touch —
    # a "caller_location" may point here without violating the drift rule's citation requirement.
    # list of {"symbol": str, "path": str, "line": int, "snippet": str}
    external_callers: list[dict] = None
    # Paths changed by this PR — used to validate/force post_inline on holistic findings.
    changed_paths: frozenset[str] = frozenset()


def build_digest(pr_meta: Any, per_file_findings: list[Finding]) -> PRDigest:
    """Assemble a compact PR digest from PRMeta and post-critic per-file findings."""
    changed_functions: list[dict] = []
    mod_funcs_by_path = pr_meta.mod_funcs_by_path or {}
    for path, funcs in mod_funcs_by_path.items():
        path_had_entry = False
        for f in funcs or []:
            name = getattr(f, "name", None) or "?"
            start = getattr(f, "start_line", 0)
            end = getattr(f, "end_line", 0)
            changed_functions.append(
                {"path": path, "name": name, "lines": f"{start}-{end}"}
            )
            path_had_entry = True
            if len(changed_functions) >= _MAX_CHANGED_FUNCTIONS:
                break
        # Always record the path even when tree-sitter couldn't extract functions
        # (e.g. eval harness with no file content). This keeps _should_skip_holistic
        # from incorrectly treating a multi-file diff as a single-file PR.
        if not path_had_entry:
            changed_functions.append({"path": path, "name": "?", "lines": "?"})
        if len(changed_functions) >= _MAX_CHANGED_FUNCTIONS:
            break

    import_edges: list[dict] = []
    import_graph = pr_meta.import_graph or {}
    for importer, callers in import_graph.items():
        for ci in callers or []:
            import_edges.append(
                {
                    "importer": importer,
                    "imports_from": ci.changed_path,
                    "symbols": list(ci.function_names),
                }
            )

    sorted_findings = sorted(
        per_file_findings,
        key=lambda f: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3, "trivial": 4}.get(
                f.impact.value, 9
            ),
            f.line,
        ),
    )
    per_file_digest: list[dict] = []
    for f in sorted_findings[:_MAX_PER_FILE_DIGEST]:
        per_file_digest.append(
            {
                "path": f.path,
                "line": f.line,
                "title": (f.title or "").strip() or _title_from_body(f.body),
                "impact": f.impact.value,
                "category": f.category,
            }
        )

    # Build diff excerpts (added lines only) for paths where function extraction failed.
    # Capped at 30 lines per file to keep the prompt token budget bounded.
    _MAX_EXCERPT_LINES = 30
    diff_excerpts: list[dict] = []
    stub_paths = {cf["path"] for cf in changed_functions if cf.get("name") == "?"}
    raw_diffs = getattr(pr_meta, "raw_diffs", None) or {}
    for path, file_diff in raw_diffs.items():
        if path not in stub_paths:
            continue
        added: list[str] = []
        for line in file_diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:])
                if len(added) >= _MAX_EXCERPT_LINES:
                    break
        if added:
            diff_excerpts.append({"path": path, "added_lines": "\n".join(added)})

    changed_paths = frozenset(cf["path"] for cf in changed_functions)
    external_callers = _collect_external_callers(pr_meta, changed_functions, changed_paths)

    return PRDigest(
        title=pr_meta.title or "",
        body=(pr_meta.body or "")[:300],
        changed_functions=changed_functions,
        import_edges=import_edges,
        per_file_findings=per_file_digest,
        diff_excerpts=diff_excerpts,
        external_callers=external_callers,
        changed_paths=changed_paths,
    )


def _collect_external_callers(
    pr_meta: Any, changed_functions: list[dict], changed_paths: frozenset[str]
) -> list[dict]:
    """Resolve real usages of changed symbols in files this PR did NOT touch, via find_callers.

    This is what lets the holistic pass flag a stale caller/implementer OUTSIDE the diff (e.g.
    an interface method's signature changed in one file, but a Lark/Office365-style implementer
    in an unchanged file still declares the old shape) — without it, the model can only ever see
    the changed files in `digest`, so it would be forced to hallucinate any unchanged-file claim.
    Best-effort and non-fatal: no repo_root, or a git-grep failure, just yields no external callers.
    """
    repo_root = getattr(pr_meta, "repo_root", None)
    if not repo_root:
        return []
    from sift.core.code_intel import find_callers

    symbols: list[str] = []
    seen_symbols: set[str] = set()
    for cf in changed_functions:
        name = cf.get("name")
        if not name or name == "?" or name in seen_symbols or not name.isidentifier():
            continue
        seen_symbols.add(name)
        symbols.append(name)
        if len(symbols) >= _MAX_EXTERNAL_CALLER_SYMBOLS:
            break

    external: list[dict] = []
    seen_locations: set[tuple[str, int]] = set()
    for symbol in symbols:
        if len(external) >= _MAX_EXTERNAL_CALLERS_TOTAL:
            break
        try:
            raw = find_callers(repo_root, symbol)
        except Exception as e:
            logger.debug("[holistic] find_callers(%s) failed: %s", symbol, e)
            continue
        if not raw or raw.startswith("["):
            continue
        per_symbol = 0
        for line in raw.splitlines():
            if per_symbol >= _MAX_EXTERNAL_CALLERS_PER_SYMBOL or len(external) >= _MAX_EXTERNAL_CALLERS_TOTAL:
                break
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            path, lineno_str, snippet = parts
            if path in changed_paths or not lineno_str.isdigit():
                continue
            loc = (path, int(lineno_str))
            if loc in seen_locations:
                continue
            seen_locations.add(loc)
            external.append({"symbol": symbol, "path": path, "line": loc[1], "snippet": snippet.strip()})
            per_symbol += 1
    return external


def _title_from_body(body: str) -> str:
    for line in (body or "").splitlines():
        line = line.strip()
        if line and not line.startswith("!["):
            return line[:80]
    return "Issue"


def _format_digest(digest: PRDigest) -> str:
    lines = [f"PR: {digest.title}"]
    if digest.body:
        lines.append(digest.body)
    lines.append("")
    lines.append(f"Changed functions ({len(digest.changed_functions)}):")
    for cf in digest.changed_functions:
        if cf.get("name") != "?":
            lines.append(f"- {cf['path']}  {cf['name']}(lines {cf['lines']})")
        else:
            lines.append(f"- {cf['path']}  (see code excerpt below)")
    if not digest.changed_functions:
        lines.append("- (none)")
    lines.append("")
    lines.append("Import edges:")
    if digest.import_edges:
        for edge in digest.import_edges:
            syms = ", ".join(edge["symbols"]) if edge["symbols"] else "?"
            lines.append(
                f"- {edge['importer']} imports {edge['imports_from']} (symbols: {syms})"
            )
    else:
        lines.append("- (none)")
    if digest.diff_excerpts:
        lines.append("")
        lines.append("Changed code (new lines added in this PR):")
        for excerpt in digest.diff_excerpts:
            lines.append(f"--- {excerpt['path']} ---")
            lines.append(excerpt["added_lines"])
    if digest.external_callers:
        lines.append("")
        lines.append(
            "Callers/co-definers of a changed symbol found OUTSIDE this PR's changed files "
            "(verified via find_callers — real matches, not a guess). `caller_location` may cite "
            "one of these path:line pairs even though the file itself was not changed by this PR; "
            "do not cite any other unchanged-file location that isn't listed here:"
        )
        for ext in digest.external_callers:
            lines.append(f"- {ext['path']}:{ext['line']} (uses `{ext['symbol']}`)  {ext['snippet']}")
    lines.append("")
    lines.append("Already found (do not repeat):")
    if digest.per_file_findings:
        for pf in digest.per_file_findings:
            lines.append(
                f"- {pf['path']}:{pf['line']}  {pf['category']}/{pf['impact']}  "
                f"{pf['title']}"
            )
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def _parse_impact(value: Any) -> Impact:
    if not value:
        return Impact.MEDIUM
    try:
        return Impact(str(value).lower())
    except ValueError:
        return Impact.MEDIUM


def _parse_certainty(value: Any) -> Certainty:
    if not value:
        return Certainty.LIKELY
    try:
        return Certainty(str(value).lower())
    except ValueError:
        return Certainty.LIKELY


def _format_holistic_body(
    item: dict,
    impact: Impact,
    certainty: Certainty,
    category: str,
) -> str:
    severity = derive_severity(impact, certainty, category)
    from sift.intelligence.llm_client import _format_structured_comment_body

    legacy = {
        "severity": severity,
        "title": item.get("title") or "Issue",
        "body": item.get("body") or "",
        "fix": item.get("fix"),
    }
    return _format_structured_comment_body(legacy)


def _loc_path(location: str) -> str:
    """'path:line' -> 'path' (path may itself contain ':' on Windows-style input; line is last)."""
    return location.rsplit(":", 1)[0] if ":" in location else location


def _parse_holistic_response(raw: str, digest: Optional[PRDigest] = None) -> list[Finding]:
    """Parse the holistic LLM response into Findings.

    `digest` (when given) lets contract-drift findings cite a stale caller/implementer OUTSIDE
    this PR's changed files — but only one the orchestrator itself verified via find_callers
    (digest.external_callers), never one the model merely asserts. Two separate guards enforce
    this against hallucination:
      1. def_location's path must be a file THIS PR changed (only a changed definition is real drift).
      2. caller_location's path must be a changed file OR one of the verified external_callers —
         any other unchanged-file claim is dropped.
    A finding whose own `path` isn't a changed file (i.e. the stale caller lives outside the diff)
    is forced to post_inline=False regardless of what the model said: there is no diff line to
    anchor an inline GitHub comment to, so it must go through the summary's off-diff routing instead.
    """
    arr = _extract_json_array(raw or "")
    if not arr:
        return []
    changed_paths = digest.changed_paths if digest else frozenset()
    external_locs = {(e["path"], e["line"]) for e in ((digest.external_callers if digest else None) or [])}
    findings: list[Finding] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        path = (item.get("path") or "").strip()
        if not path:
            continue
        try:
            line = int(item.get("line"))
        except (TypeError, ValueError):
            continue
        if line <= 0:
            continue
        # Contract-drift only: require both anchors (changed definition + stale caller).
        # A finding that can't cite both is an opinion, not a drift bug — drop it.
        def_loc = (item.get("def_location") or "").strip()
        caller_loc = (item.get("caller_location") or "").strip()
        if ":" not in def_loc or ":" not in caller_loc:
            logger.debug(
                "[holistic] drop finding without two-sided citation: %r", item.get("title")
            )
            continue
        if changed_paths and _loc_path(def_loc) not in changed_paths:
            logger.debug(
                "[holistic] drop finding: def_location %r is not a file changed by this PR", def_loc
            )
            continue
        if changed_paths:
            caller_path = _loc_path(caller_loc)
            try:
                caller_line = int(caller_loc.rsplit(":", 1)[1])
            except (IndexError, ValueError):
                caller_line = None
            caller_is_changed = caller_path in changed_paths
            caller_is_verified_external = (caller_path, caller_line) in external_locs
            if not (caller_is_changed or caller_is_verified_external):
                logger.debug(
                    "[holistic] drop finding: caller_location %r is neither a changed file nor "
                    "a find_callers-verified external location", caller_loc
                )
                continue
        # Surface both anchors in the posted comment.
        item = {
            **item,
            "body": f"{(item.get('body') or '').strip()}\n\n"
                    f"Contract drift — changed definition: {def_loc}; stale caller: {caller_loc}.",
        }
        impact = _parse_impact(item.get("impact"))
        certainty = _parse_certainty(item.get("certainty"))
        # Contract drift is a runtime break: default to correctness (survives the
        # design/maintainability gate) rather than 'design'.
        category = (item.get("category") or "correctness").lower()
        if category not in CATEGORIES:
            category = "correctness"
        post_inline = item.get("post_inline", True)
        if isinstance(post_inline, str):
            post_inline = post_inline.lower() not in ("false", "0", "no")
        # An unchanged file has no diff line to anchor an inline comment to — force the summary
        # route regardless of what the model said. (Safe even when changed_paths is unavailable:
        # in that case we simply trust the model's post_inline, same as before this change.)
        if changed_paths and path not in changed_paths:
            post_inline = False
        findings.append(
            Finding(
                path=path,
                line=line,
                title=(item.get("title") or "").strip() or "Issue",
                body=_format_holistic_body(item, impact, certainty, category),
                impact=impact,
                certainty=certainty,
                category=category,
                origin="holistic",
                fix=(item.get("fix") or None),
                post_inline=bool(post_inline),
            )
        )
    return findings


def _should_skip_holistic(digest: PRDigest) -> bool:
    """Skip when there are no real cross-file relationships to reason about.

    Multiple files in a diff is not sufficient — the holistic pass is only
    meaningful when files actually import each other (import_edges) or when
    the same function appears across files (suggesting interface drift).
    Without import edges, the pass generates spurious cross-file findings.
    """
    if not digest.import_edges:
        return True
    paths = {cf["path"] for cf in digest.changed_functions}
    if len(paths) < 2:
        return True
    return False


async def review_holistic(
    digest: PRDigest,
    plan: EffortPlan,
    cap: ModelCapability,
) -> list[Finding]:
    """Run whole-PR holistic pass; returns cross-file findings."""
    _ = plan
    _ = cap
    if _should_skip_holistic(digest):
        logger.info(
            "[holistic] skipped: import_edges=%d, files with changed functions=%d "
            "(needs PR-internal imports across 2+ changed files)",
            len(digest.import_edges),
            len({cf["path"] for cf in digest.changed_functions}),
        )
        return []

    user_content = _format_digest(digest)
    try:
        raw = await _call_llm(
            HOLISTIC_SYSTEM,
            user_content,
            model=config.SIFT_REVIEW_MODEL or config.LLM_MODEL,
            api_base=config.SIFT_REVIEW_MODEL_BASE_URL or config.LLM_API_BASE or None,
            api_key=config.SIFT_REVIEW_MODEL_KEY or None,
        )
    except Exception as e:
        logger.warning("[holistic] LLM call failed: %s", e)
        return []

    findings = _parse_holistic_response(raw or "", digest)
    logger.info("[holistic] %d finding(s) from digest", len(findings))
    return findings
