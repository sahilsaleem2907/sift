"""Promote ERROR-severity static-tool findings directly to Finding objects.

Static tools (semgrep, gitleaks via semgrep rules, codeql) that fire at ERROR
severity are treated as confirmed — their reporting must not depend on LLM
cooperation. This module:

1. Converts each auto-promotable tool finding to a Finding with
   critic_exempt=True (critic and noise gate skip it).
2. Optionally enriches the body/fix by calling the LLM once per batch —
   but the LLM call can ONLY improve the text; it cannot drop or downgrade.
   On any parse failure the raw tool message is used as-is.
"""
from __future__ import annotations

import logging
from typing import Any

from src import config
from src.intelligence.llm_client import (
    _call_llm,
    _extract_json_array,
    _format_structured_comment_body,
)
from src.intelligence.schema import Certainty, Finding, Impact

logger = logging.getLogger(__name__)

# Semgrep/codeql severity strings that trigger auto-promotion
_AUTO_PROMOTE_SEVERITIES = frozenset({"ERROR"})

# Rule-id substrings that signal a secret / credential finding regardless of severity
_SECRET_RULE_SUBSTRINGS = (
    "secret", "token", "credential", "api-key", "apikey",
    "password", "private-key", "gitleaks", "hardcoded",
)

_ENRICH_SYSTEM = """You are a code review assistant. You are given a list of security or
error findings detected by a static analysis tool. For each finding, write a concise,
developer-friendly title, explanation, and (where obvious) a suggested fix.

Rules:
- Do NOT change the severity, category, or verdict — the tool has already confirmed these.
- Do NOT drop any finding. You must return one entry per input finding.
- Title: max 5 words, specific to the finding (e.g. "Hardcoded GitHub token", "Syntax error in registerCommand").
- Keep body to 1-3 sentences. Fix is optional; omit if no clean fix is obvious.

Respond with a JSON array. One object per input finding, in the same order:
{
  "index": <0-based integer>,
  "title": "<max 5 word title>",
  "body": "<improved explanation>",
  "fix": "<suggested fix or empty string>"
}
No markdown fences. No prose outside the array."""


def _is_secret_rule(rule_id: str) -> bool:
    low = rule_id.lower()
    return any(s in low for s in _SECRET_RULE_SUBSTRINGS)


def _tool_finding_impact(f: dict) -> Impact:
    """Map tool severity + rule to Impact."""
    sev = (f.get("severity") or "").upper()
    rule_id = f.get("check_id") or f.get("rule_id") or ""
    if _is_secret_rule(rule_id):
        return Impact.CRITICAL
    if sev == "ERROR":
        return Impact.HIGH
    return Impact.MEDIUM


def _tool_finding_category(f: dict) -> str:
    rule_id = (f.get("check_id") or f.get("rule_id") or "").lower()
    if _is_secret_rule(rule_id) or "injection" in rule_id or "sqli" in rule_id:
        return "security"
    if "perf" in rule_id or "performance" in rule_id:
        return "perf"
    return "correctness"


def _raw_body(f: dict, origin: str) -> str:
    rule_id = f.get("check_id") or f.get("rule_id") or ""
    msg = (f.get("message") or "").strip()
    suffix = " [FILE-WIDE]" if f.get("critical_bypass") else ""
    return f"[{origin.upper()}] {rule_id}: {msg}{suffix}"


def should_auto_promote(f: dict) -> bool:
    """Return True if this tool finding must be auto-promoted (guaranteed output)."""
    sev = (f.get("severity") or "").upper()
    rule_id = f.get("check_id") or f.get("rule_id") or ""
    return sev in _AUTO_PROMOTE_SEVERITIES or _is_secret_rule(rule_id)


def _build_finding(f: dict, path: str, origin: str, body: str, fix: str, title: str = "") -> Finding:
    impact = _tool_finding_impact(f)
    category = _tool_finding_category(f)
    # Use enriched title if provided; fall back to rule-ID last segment.
    rule_id_title = (f.get("check_id") or f.get("rule_id") or origin).split(".")[-1][:60]
    title = (title.strip() or rule_id_title)[:60]

    # Derive the severity label so promoted findings get the same badge as LLM
    # findings — security → SECURITY, high correctness → BUG, etc.
    from src.intelligence.schema import derive_severity
    severity = derive_severity(impact, Certainty.CONFIRMED, category)
    badged_body = _format_structured_comment_body({
        "severity": severity,
        "title": title,
        "body": body,
        "fix": fix or "",
    })

    return Finding(
        path=path,
        line=int(f.get("line") or 1),
        title=title,
        body=badged_body,
        impact=impact,
        certainty=Certainty.CONFIRMED,
        category=category,
        origin=origin,
        fix=fix or None,
        post_inline=True,
        critic_exempt=True,
    )


async def _enrich_batch(
    raw_findings: list[dict],
    path: str,
    origin: str,
    diff: str,
) -> list[dict]:
    """Call LLM once to improve title/body/fix for a batch. Returns list of dicts.

    Each dict has keys: "body", "fix", "title".
    Falls back to raw tool messages on any failure — reporting is never blocked.
    """
    _fallback = [{"body": _raw_body(f, origin), "fix": "", "title": ""} for f in raw_findings]
    if not config.LLM_MODEL:
        return _fallback

    items = "\n".join(
        f'[{i}] rule={f.get("check_id") or f.get("rule_id") or ""} '
        f'line={f.get("line")} message={f.get("message") or ""}'
        for i, f in enumerate(raw_findings)
    )
    user_content = (
        f"File: {path}\n\nDiff (for context):\n{diff[:2000]}\n\n"
        f"Tool findings to enrich:\n{items}"
    )

    try:
        raw = await _call_llm(
            _ENRICH_SYSTEM,
            user_content,
            model=config.SIFT_REVIEW_MODEL or config.LLM_MODEL,
            api_base=config.SIFT_REVIEW_MODEL_BASE_URL or config.LLM_API_BASE or None,
            api_key=config.SIFT_REVIEW_MODEL_KEY or None,
        )
    except Exception as exc:
        logger.warning("[static_promote] enrich LLM call failed (%s); using raw messages", exc)
        return _fallback

    # Use the shared hardened extractor — strips reasoning blocks and survives
    # prose/markdown around the array (same failure mode that broke candidates).
    parsed = _extract_json_array(raw)
    if parsed is None:
        logger.warning(
            "[static_promote] enrich parse FAILURE for %s: %d chars received but no "
            "JSON array extracted; using raw tool messages. Raw head: %r",
            path, len(raw or ""), (raw or "").strip()[:300],
        )
        return _fallback
    index_map = {int(e["index"]): e for e in parsed if isinstance(e, dict) and "index" in e}

    result = []
    for i, f in enumerate(raw_findings):
        entry = index_map.get(i)
        if entry:
            body = (entry.get("body") or "").strip() or _raw_body(f, origin)
            fix = (entry.get("fix") or "").strip()
            title = (entry.get("title") or "").strip()[:60]
        else:
            body = _raw_body(f, origin)
            fix = ""
            title = ""
        result.append({"body": body, "fix": fix, "title": title})
    return result


async def promote_static_findings(
    path: str,
    file_diff: str,
    semgrep_findings: list[dict],
    codeql_findings: list[dict],
    pyright_findings: list[dict] | None = None,
    analyzer_findings: list[dict] | None = None,
) -> list[Finding]:
    """Convert ERROR/secret tool findings to critic_exempt Findings, enriched by LLM.

    Returns findings guaranteed to appear in the output regardless of what the
    LLM generation pass finds or drops.
    """
    to_promote: list[tuple[dict, str]] = []  # (raw_finding, origin)
    for f in semgrep_findings:
        if should_auto_promote(f):
            to_promote.append((f, "semgrep"))
    for f in codeql_findings:
        if should_auto_promote(f):
            to_promote.append((f, "codeql"))
    for f in pyright_findings or []:
        if should_auto_promote(f):
            to_promote.append((f, "pyright"))
    for f in analyzer_findings or []:
        if should_auto_promote(f):
            to_promote.append((f, "analyzer"))

    if not to_promote:
        return []

    logger.info(
        "[static_promote] path=%s: auto-promoting %d finding(s) "
        "(%s semgrep, %s codeql, %s pyright, %s analyzer)",
        path,
        len(to_promote),
        sum(1 for _, o in to_promote if o == "semgrep"),
        sum(1 for _, o in to_promote if o == "codeql"),
        sum(1 for _, o in to_promote if o == "pyright"),
        sum(1 for _, o in to_promote if o == "analyzer"),
    )

    # Group by origin for batched enrichment calls
    enriched: dict[int, dict] = {}
    for origin in ("semgrep", "codeql", "pyright", "analyzer"):
        batch = [(f, i) for i, (f, o) in enumerate(to_promote) if o == origin]
        if not batch:
            continue
        raw_fs = [f for f, _ in batch]
        enrich_results = await _enrich_batch(raw_fs, path, origin, file_diff)
        for (_, idx), entry in zip(batch, enrich_results):
            enriched[idx] = entry

    findings = []
    for idx, (f, origin) in enumerate(to_promote):
        entry = enriched.get(idx, {"body": _raw_body(f, origin), "fix": "", "title": ""})
        findings.append(_build_finding(
            f, path, origin,
            body=entry["body"],
            fix=entry["fix"],
            title=entry.get("title", ""),
        ))
        logger.debug(
            "[static_promote] promoted line=%s impact=%s category=%s origin=%s rule=%s",
            f.get("line"), findings[-1].impact.value, findings[-1].category,
            origin, f.get("check_id") or f.get("rule_id") or "",
        )

    return findings
