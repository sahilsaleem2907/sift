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


@dataclass
class PRDigest:
    title: str
    body: str
    changed_functions: list[dict]
    import_edges: list[dict]
    per_file_findings: list[dict]
    # raw added lines per path; shown in digest when function extraction is unavailable
    diff_excerpts: list[dict] = None  # list of {"path": str, "added_lines": str}


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

    return PRDigest(
        title=pr_meta.title or "",
        body=(pr_meta.body or "")[:300],
        changed_functions=changed_functions,
        import_edges=import_edges,
        per_file_findings=per_file_digest,
        diff_excerpts=diff_excerpts,
    )


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


def _parse_holistic_response(raw: str) -> list[Finding]:
    arr = _extract_json_array(raw or "")
    if not arr:
        return []
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
        impact = _parse_impact(item.get("impact"))
        certainty = _parse_certainty(item.get("certainty"))
        category = (item.get("category") or "design").lower()
        if category not in CATEGORIES:
            category = "design"
        post_inline = item.get("post_inline", True)
        if isinstance(post_inline, str):
            post_inline = post_inline.lower() not in ("false", "0", "no")
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

    findings = _parse_holistic_response(raw or "")
    logger.info("[holistic] %d finding(s) from digest", len(findings))
    return findings
