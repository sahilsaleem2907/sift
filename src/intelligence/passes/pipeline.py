"""Review pipeline orchestrator.

Phase 3: per-file candidates (+critic) then holistic pass + severity gate.
"""
import logging
from dataclasses import dataclass
from typing import Any, Optional

from src import config
from src.intelligence.capability import ModelCapability
from src.intelligence.effort import EffortPlan
from src.intelligence.passes.candidates import generate_candidates
from src.intelligence.schema import Finding

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


async def run_pipeline_per_file(
    file: FileReviewInput,
    pr_title: str,
    plan: EffortPlan,
    cap: ModelCapability,
) -> list[Finding]:
    """Per-file candidates and optional critic (no holistic, no severity gate)."""
    from src.intelligence.passes.critic import critique, rule_dedupe

    candidates = await generate_candidates(file.file_diff, file.path, file.pr_context)
    logger.debug("[pipeline] %s: %d candidate(s)", file.path, len(candidates))

    use_llm_critic = plan.run_critic and candidates and bool(config.SIFT_REVIEW_MODEL)
    if use_llm_critic:
        candidates = await critique(
            candidates, file.file_diff, pr_title, plan, cap
        )
        logger.debug("[pipeline] %s: %d after critic", file.path, len(candidates))
    elif plan.run_critic and candidates and not config.SIFT_REVIEW_MODEL:
        logger.debug(
            "[pipeline] %s: SIFT_REVIEW_MODEL not set, using rule_dedupe",
            file.path,
        )

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
    from src.intelligence.passes.critic import rule_dedupe
    from src.intelligence.passes.holistic import build_digest, review_holistic
    from src.intelligence.passes.severity import apply_severity_gate

    all_findings = list(all_per_file_findings)

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

    return apply_severity_gate(all_findings, plan)


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
        per_file.extend(await run_pipeline_per_file(f, pr_title, plan, cap))
    return await run_pipeline_holistic(per_file, pr_meta, plan, cap)
