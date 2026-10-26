# SPDX-License-Identifier: AGPL-3.0-only
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
    # Filesystem path to the PR's git checkout; powers repo-wide fact-tools.
    repo_root: Optional[str] = None


def _in_range(line: Any, start: int, end: int) -> bool:
    return isinstance(line, int) and start <= line <= end


def scope_context_to_lines(
    pr_context: dict[str, Any],
    path: str,
    chunk_diff: str,
    start: int,
    end: int,
    mod_funcs: list[Any],
    first_chunk: bool,
) -> dict[str, Any]:
    """Restrict the line-anchored parts of a file's context to new-file lines [start, end].

    Static-tool findings, file-context ranges, AST nodes and the before/after of changed
    functions are kept only for this chunk; file-wide findings ([FILE-WIDE]) go to the first
    chunk. File-level blocks (callers, callees, similar snippets) are kept as-is — the
    review_file budget drops them first anyway.
    """
    from sift.intelligence.ast.diff_ast import filter_ast_to_range
    from sift.intelligence.retrieval import _semantic_before_after

    def keep(items: Any) -> list:
        return [
            i for i in (items or [])
            if _in_range(i.get("line"), start, end) or (first_chunk and i.get("critical_bypass"))
        ]

    out = dict(pr_context)
    for key in ("semgrep_findings", "codeql_findings", "linter_issues"):
        if key in out:
            out[key] = keep(out[key])
    fc = out.get("file_context")
    if isinstance(fc, dict):
        ranges = [
            (max(s, start), min(e, end)) for s, e in fc.get("ranges") or [] if e >= start and s <= end
        ]
        out["file_context"] = {**fc, "ranges": ranges or [(start, end)]}
    if out.get("ast_diff"):
        out["ast_diff"] = filter_ast_to_range(out["ast_diff"], start, end)
    if out.get("semantic_before_after"):
        funcs = [f for f in mod_funcs if f.end_line >= start and f.start_line <= end]
        out["semantic_before_after"] = _semantic_before_after(path, chunk_diff, funcs)
    return out


def _review_window(cap: ModelCapability, plan: EffortPlan) -> int:
    """Context window a chunk must fit: the smaller of the generator's and the critic's."""
    window = cap.context_window
    if plan.run_critic:
        from sift.intelligence.capability import review_capability

        window = min(window, review_capability().context_window)
    return window


def _plan_chunks(
    file: FileReviewInput,
    enriched: dict[str, Any],
    window: int,
    reasoning: bool,
    mod_funcs: list[Any],
) -> list[Any]:
    """The diff chunks to review: one (the whole diff) unless even the smallest prompt overflows."""
    from sift.intelligence.budget import DiffChunk, count_tokens, input_budget, split_diff
    from sift.intelligence.llm_client import min_review_prompt_tokens

    budget = input_budget(window, reasoning)
    min_total, diff_block_tokens = min_review_prompt_tokens(
        file.file_diff, file.path, enriched, reasoning
    )
    if min_total <= budget:
        return [DiffChunk(file.file_diff, 0, 0)]
    # Diff tokens available per chunk, converted from the annotated diff block (which carries
    # [L<n>] markers) to the raw diff that split_diff measures. split_diff sizes chunks by the
    # whole diff's average chars/token, so leave 10% for chunks denser than average.
    fixed = min_total - diff_block_tokens
    raw_tokens = count_tokens(file.file_diff)
    per_chunk = int(max(1024, budget - fixed) * 0.9) * raw_tokens // max(1, diff_block_tokens)
    chunks = split_diff(
        file.file_diff,
        per_chunk,
        func_ranges=[(f.start_line, f.end_line) for f in mod_funcs],
    )
    logger.info(
        "[budget] %s: diff exceeds the %d-token input budget (min prompt %d); split into %d chunk(s)",
        file.path, budget, min_total, len(chunks),
    )
    return chunks


async def run_pipeline_per_file(
    file: FileReviewInput,
    pr_title: str,
    plan: EffortPlan,
    cap: ModelCapability,
    pr_meta: Optional[PRMeta] = None,
) -> list[Finding]:
    """Per-file candidates and optional critic (no holistic, no severity gate).

    A diff too large for the model's context window is reviewed in chunks (see _plan_chunks).
    If the provider still rejects a prompt as too long, the file is re-planned once with the
    budget scaled down by the overflow the provider reported.
    """
    from sift.intelligence.budget import overflow_shrink_factor
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
    pyright_raw = (file.pr_context or {}).get("pyright_findings") or []
    analyzer_raw = (file.pr_context or {}).get("analyzer_findings") or []
    promoted = await promote_static_findings(
        file.path, file.file_diff, semgrep_raw, codeql_raw, pyright_raw, analyzer_raw
    )
    if promoted:
        logger.info("[pipeline] %s: %d static finding(s) auto-promoted", file.path, len(promoted))

    window = _review_window(cap, plan)
    reasoning = bool(plan.reasoning_effort)
    mod_funcs = list(mod_funcs_by_path.get(file.path) or [])
    candidates: list[Finding] = []
    for attempt in range(2):
        try:
            chunks = _plan_chunks(file, enriched, window, reasoning, mod_funcs)
            candidates = []
            for idx, chunk in enumerate(chunks):
                candidates.extend(await _review_chunk(
                    file, chunk, chunks, idx, enriched, retrieval_ctx, pr_title, plan, cap,
                    window, pr_meta, path_to_content, mod_funcs_by_path, mod_funcs,
                ))
            break
        except Exception as e:
            shrink = overflow_shrink_factor(e)
            if shrink is None or attempt == 1:
                raise
            window = int(window * shrink)
            logger.warning(
                "[budget] %s: provider rejected the prompt as too long; re-planning with a "
                "%d-token window (%s)", file.path, window, e,
            )

    # Merge promoted static findings in so rule_dedupe can resolve any overlap between
    # static and LLM findings on the same line.
    candidates = promoted + candidates

    # Always collapse duplicate (path, line) findings, keeping the highest impact.
    # The LLM critic verifies/re-rates but does not dedupe, so a generator that emits
    # two findings on the same line would otherwise survive as duplicates.
    from sift.intelligence.passes.critic import rule_dedupe
    return rule_dedupe(candidates)


async def _review_chunk(
    file: FileReviewInput,
    chunk: Any,
    chunks: list[Any],
    idx: int,
    enriched: dict[str, Any],
    retrieval_ctx: Any,
    pr_title: str,
    plan: EffortPlan,
    cap: ModelCapability,
    window: int,
    pr_meta: Optional[PRMeta],
    path_to_content: dict[str, str],
    mod_funcs_by_path: dict,
    mod_funcs: list[Any],
) -> list[Finding]:
    """Generate candidates (+critic) for one diff chunk; a single chunk is the whole file."""
    from sift.intelligence.passes.critic import critique

    chunked = len(chunks) > 1
    diff = chunk.text
    ctx = enriched
    if chunked:
        ctx = scope_context_to_lines(
            enriched, file.path, diff, chunk.new_start, chunk.new_end, mod_funcs, idx == 0
        )

    if plan.enable_agentic and cap.supports_function_calling:
        from dataclasses import replace

        from sift.intelligence.passes.agentic import agentic_review
        from sift.intelligence.retrieval import _assemble_static_tools

        chunk_retrieval = retrieval_ctx
        if chunked:
            chunk_retrieval = replace(
                retrieval_ctx,
                diff=diff,
                semantic_before_after=ctx.get("semantic_before_after") or "",
                static_tools=_assemble_static_tools(ctx),
            )
        candidates = await agentic_review(
            FileReviewInput(file.path, diff, ctx),
            plan,
            cap,
            path_to_content,
            mod_funcs_by_path,
            chunk_retrieval,
            repo_root=(pr_meta.repo_root if pr_meta else None),
            context_window=window,
        )
    else:
        if plan.enable_agentic:
            logger.info(
                "[pipeline] %s: agentic skipped (model reports no tool calling); one-shot review",
                file.path,
            )
        # Request reasoning on the primary generator only when the model supports it.
        reasoning_effort = plan.reasoning_effort if cap.supports_reasoning else None
        candidates = await generate_candidates(
            diff, file.path, ctx, reasoning_effort=reasoning_effort, context_window=window
        )
    if chunked:
        # A chunk only saw its own lines: drop findings it placed on another chunk's lines.
        # Lines outside every chunk (e.g. a [FILE-WIDE] tool finding) are kept and routed
        # off-diff as usual.
        others = [c for j, c in enumerate(chunks) if j != idx]
        candidates = [
            f for f in candidates
            if _in_range(f.line, chunk.new_start, chunk.new_end)
            or not any(_in_range(f.line, c.new_start, c.new_end) for c in others)
        ]
    logger.debug("[pipeline] %s: %d candidate(s) (chunk %d/%d)", file.path, len(candidates), idx + 1, len(chunks))

    # Run critic when plan calls for it — falls back to primary model when no
    # separate SIFT_REVIEW_MODEL is configured (previously this was silently skipped).
    if plan.run_critic and candidates:
        if not config.SIFT_REVIEW_MODEL:
            logger.debug(
                "[pipeline] %s: SIFT_REVIEW_MODEL not set, critic using primary model",
                file.path,
            )
        critic_ctx = {
            **ctx,
            # Carried for the symmetric critic tool loop (high effort).
            "_repo_root": (pr_meta.repo_root if pr_meta else None),
            "_path_to_content": path_to_content,
            "_mod_funcs_by_path": mod_funcs_by_path,
        }
        candidates = await critique(candidates, diff, pr_title, plan, cap, critic_ctx)
        logger.debug("[pipeline] %s: %d after critic", file.path, len(candidates))
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
