"""Review pipeline orchestrator.

Phase 3: per-file candidates (+critic) then holistic pass + severity gate.
"""
import logging
from dataclasses import dataclass
from typing import Any, Optional

from sift import config
from sift.intelligence.capability import ModelCapability
from sift.intelligence.effort import EffortPlan
from sift.intelligence.passes.candidates import generate_candidates
from sift.intelligence.schema import Finding

logger = logging.getLogger(__name__)


@dataclass
class FileReviewInput:
    path: str
    file_diff: str
    pr_context: dict[str, Any]


@dataclass
class PRMeta:
    title: str
    body: str
    import_graph: Optional[dict] = None
    mod_funcs_by_path: Optional[dict] = None
    # path -> unified diff text; used by holistic pass for code context when tree-sitter
    # extraction is unavailable (e.g. eval harness without full file content).
    raw_diffs: Optional[dict] = None
    path_to_content: Optional[dict] = None


async def run_pipeline_per_file(
    file: FileReviewInput,
    pr_title: str,
    plan: EffortPlan,
    cap: ModelCapability,
    pr_meta: Optional[PRMeta] = None,
) -> list[Finding]:
    """Per-file candidates and optional critic (no holistic, no severity gate)."""
    from sift.intelligence.passes.critic import critique, rule_dedupe
    from sift.intelligence.retrieval import build_context

    path_to_content = (pr_meta.path_to_content if pr_meta else None) or {}
    mod_funcs_by_path = (pr_meta.mod_funcs_by_path if pr_meta else None) or {}
    import_graph = (pr_meta.import_graph if pr_meta else None) or {}

    retrieval_ctx = build_context(
        file.path,
        file.file_diff,
        file.pr_context,
        plan,
        cap,
        path_to_content,
        mod_funcs_by_path,
        import_graph,
    )
    enriched = {**(file.pr_context or {}), **retrieval_ctx.to_pr_context_dict()}

    # Auto-promote ERROR/secret static-tool findings before LLM generation
    from sift.intelligence.passes.static_promote import promote_static_findings
    semgrep_raw = (file.pr_context or {}).get("semgrep_findings") or []
    codeql_raw = (file.pr_context or {}).get("codeql_findings") or []
    promoted = await promote_static_findings(file.path, file.file_diff, semgrep_raw, codeql_raw)
    if promoted:
        logger.info("[pipeline] %s: %d static finding(s) auto-promoted", file.path, len(promoted))

    if plan.enable_agentic and cap.supports_function_calling:
        from sift.intelligence.passes.agentic import agentic_review

        candidates = await agentic_review(
            FileReviewInput(file.path, file.file_diff, enriched),
            plan,
            cap,
            path_to_content,
            mod_funcs_by_path,
            retrieval_ctx,
        )
    else:
        candidates = await generate_candidates(file.file_diff, file.path, enriched)
    logger.debug("[pipeline] %s: %d candidate(s) + %d promoted", file.path, len(candidates), len(promoted))

    # Merge promoted static findings in before the critic so rule_dedupe can
    # resolve any overlap between static and LLM findings on the same line.
    candidates = promoted + candidates

    # Run critic when plan calls for it — falls back to primary model when no
    # separate SIFT_REVIEW_MODEL is configured (previously this was silently skipped).
    use_llm_critic = plan.run_critic and bool(candidates)
    if use_llm_critic:
        if not config.SIFT_REVIEW_MODEL:
            logger.debug(
                "[pipeline] %s: SIFT_REVIEW_MODEL not set, critic using primary model",
                file.path,
            )
        candidates = await critique(
            candidates, file.file_diff, pr_title, plan, cap
        )
        logger.debug("[pipeline] %s: %d after critic", file.path, len(candidates))

    # Always collapse duplicate (path, line) findings, keeping the highest impact.
    # The LLM critic verifies/re-rates but does not dedupe, so a generator that emits
    # two findings on the same line would otherwise survive as duplicates.
    candidates = rule_dedupe(candidates)

    return candidates


async def run_pipeline_holistic(
    all_per_file_findings: list[Finding],
    pr_meta: PRMeta,
    plan: EffortPlan,
    cap: ModelCapability,
) -> list[Finding]:
    """Holistic pass, dedupe holistic findings, severity gate."""
    from sift.intelligence.passes.critic import rule_dedupe
    from sift.intelligence.passes.holistic import build_digest, review_holistic
    from sift.intelligence.passes.severity import (
        apply_final_severity_labels,
        apply_severity_gate,
    )

    all_findings = list(all_per_file_findings)

    # Deterministic intra-PR duplicate logic detection — runs regardless of
    # import edges, no LLM required, findings are critic_exempt.
    if pr_meta.mod_funcs_by_path:
        from sift.intelligence.passes.duplicate_detect import detect_duplicate_functions
        dup_findings = await detect_duplicate_functions(pr_meta.mod_funcs_by_path)
        if dup_findings:
            logger.info("[pipeline] duplicate_detect: %d finding(s)", len(dup_findings))
            all_findings.extend(dup_findings)

    if plan.run_holistic:
        digest = build_digest(pr_meta, all_findings)
        holistic = await review_holistic(digest, plan, cap)
        per_file_keys = {(f.path, f.line, f.category) for f in all_findings}
        holistic = [
            f
            for f in holistic
            if (f.path, f.line, f.category) not in per_file_keys
        ]
        if holistic:
            logger.debug("[pipeline] holistic: %d new finding(s)", len(holistic))
            # Holistic findings skip the per-file LLM critic: that prompt is tuned for
            # single-file bugs and often drops cross-file design/maintainability issues,
            # especially when critique() is called with an empty diff. The holistic pass
            # is already a dedicated second pass for PR-wide concerns.
            holistic = rule_dedupe(holistic)
            all_findings.extend(holistic)

    return apply_final_severity_labels(apply_severity_gate(all_findings, plan))


async def run_pipeline(
    files: list[FileReviewInput],
    pr_meta: PRMeta,
    plan: EffortPlan,
    cap: ModelCapability,
) -> list[Finding]:
    """Run full pipeline: per-file stage then holistic + severity gate."""
    per_file: list[Finding] = []
    pr_title = pr_meta.title or ""
    for f in files:
        per_file.extend(await run_pipeline_per_file(f, pr_title, plan, cap, pr_meta))
    return await run_pipeline_holistic(per_file, pr_meta, plan, cap)
