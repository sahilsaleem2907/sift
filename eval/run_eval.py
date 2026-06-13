"""Offline golden-set scorer with consistency tracking.

Usage:
    # single run (original behaviour)
    python -m eval.run_eval --model ollama/llama3.2 --effort low

    # consistency mode: run each case N times, report hit-rate + flip-rate
    python -m eval.run_eval --model ollama/deepseek-coder-v2:16b --effort balanced --runs 5

    # single case, verbose findings
    python -m eval.run_eval --model anthropic/claude-opus-4-8 --effort balanced --case 011_hardcoded_secret_typo -v
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import defaultdict
from pathlib import Path

from eval.schema import ExpectedFinding, GoldenCase
from eval.secret_scan import scan_diff_for_secrets
from src import config
from src.core.import_analyzer import resolve_pr_import_graph
from src.core.pr_analyzer import split_diff_by_file
from src.intelligence.ast.function_extract import extract_modified_functions
from src.intelligence.capability import detect
from src.intelligence.effort import EffortLevel, plan_for
from src.intelligence.passes.pipeline import (
    FileReviewInput,
    PRMeta,
    run_pipeline_holistic,
    run_pipeline_per_file,
)
from src.intelligence.schema import Impact

_IMPACT_RANK = {i.value: n for n, i in enumerate(Impact)}
CASES_DIR = Path(__file__).parent / "cases"


def _is_hit(finding, expected: ExpectedFinding) -> bool:
    lo, hi = expected.line_range
    finding_rank = _IMPACT_RANK.get(finding.impact.value, 9)
    min_rank = _IMPACT_RANK.get(expected.min_impact, 9)
    return (
        lo <= finding.line <= hi
        and finding.category in expected.categories
        and finding_rank <= min_rank
    )


def _content_from_diff(file_diff: str) -> str:
    """Reconstruct approximate new-file content from a unified diff."""
    lines = []
    for line in file_diff.splitlines():
        if line.startswith(("diff ", "--- ", "+++ ", "@@ ", "index ")):
            continue
        if line.startswith("+"):
            lines.append(line[1:])
        elif line.startswith(" "):
            lines.append(line[1:])
    return "\n".join(lines)


def _build_pr_meta(case: GoldenCase) -> PRMeta:
    file_chunks = split_diff_by_file(case.diff_text)
    path_to_content: dict[str, str] = {
        path: _content_from_diff(file_diff) for path, file_diff in file_chunks
    }
    mod_funcs_by_path: dict = {}
    for path, file_diff in file_chunks:
        try:
            mod_funcs_by_path[path] = extract_modified_functions(
                path, path_to_content.get(path) or "", file_diff
            )
        except Exception:
            mod_funcs_by_path[path] = []
    import_graph = resolve_pr_import_graph(
        file_chunks, path_to_content, mod_funcs_by_path
    )
    raw_diffs = {path: file_diff for path, file_diff in file_chunks}
    return PRMeta(
        title=case.description,
        body="",
        import_graph=import_graph,
        mod_funcs_by_path=mod_funcs_by_path,
        raw_diffs=raw_diffs,
        path_to_content=path_to_content,
    )


async def score_case(case: GoldenCase, plan, cap) -> dict:
    """Score one case, returning hits/misses/noise and the raw findings."""
    file_chunks = split_diff_by_file(case.diff_text)
    if not file_chunks:
        file_chunks = [(case.path, case.diff_text)]

    all_findings = []
    pr_meta = _build_pr_meta(case)
    for path, file_diff in file_chunks:
        inp = FileReviewInput(
            path=path,
            file_diff=file_diff,
            pr_context={
                "title": case.description,
                "body": "",
                "semgrep_findings": scan_diff_for_secrets(file_diff),
            },
        )
        per_file = await run_pipeline_per_file(
            inp, case.description, plan, cap, pr_meta
        )
        all_findings.extend(per_file)
    findings = await run_pipeline_holistic(all_findings, pr_meta, plan, cap)

    hits, misses, noise = 0, 0, 0
    matched_finding_ids: set[int] = set()
    hit_expected_indices: set[int] = set()
    for exp_idx, exp in enumerate(case.expected):
        matched = [f for f in findings if _is_hit(f, exp)]
        if matched:
            hits += 1
            hit_expected_indices.add(exp_idx)
            matched_finding_ids.update(id(f) for f in matched)
        else:
            misses += 1

    for f in findings:
        if id(f) not in matched_finding_ids and f.line not in case.false_positive_lines:
            noise += 1

    return {
        "case": case.id,
        "hits": hits,
        "misses": misses,
        "noise": noise,
        "findings": len(findings),
        "hit_expected_indices": hit_expected_indices,
        "_findings": findings,
    }


def _print_single_run(results: list[dict], model: str, effort: str, verbose: bool) -> None:
    total_expected = sum(r["hits"] + r["misses"] for r in results)
    total_hits = sum(r["hits"] for r in results)
    total_noise = sum(r["noise"] for r in results)
    total_findings = sum(r["findings"] for r in results)

    precision = total_hits / total_findings if total_findings else 0.0
    recall = total_hits / total_expected if total_expected else 0.0
    noise_rate = total_noise / total_findings if total_findings else 0.0

    print(f"\nModel: {model}  Effort: {effort}")
    print(f"Precision: {precision:.0%}  Recall: {recall:.0%}  Noise-rate: {noise_rate:.0%}")
    print(f"  ({total_hits}/{total_expected} expected hits, {total_noise} noise findings)\n")
    for r in results:
        status = "OK  " if r["misses"] == 0 else "MISS"
        print(f"  [{status}] {r['case']}: hits={r['hits']} misses={r['misses']} noise={r['noise']}")
        if verbose and r["_findings"]:
            for f in r["_findings"]:
                print(
                    f"         path={f.path} line={f.line} "
                    f"impact={f.impact.value} certainty={f.certainty.value} "
                    f"category={f.category} origin={f.origin}"
                )


def _hit_rate_symbol(hit_rate: float) -> str:
    if hit_rate >= 1.0:
        return "✓"
    if hit_rate >= 0.5:
        return "~"
    return "✗"


def _print_consistency_report(
    cases: list[GoldenCase],
    all_run_results: list[list[dict]],
    model: str,
    effort: str,
    runs: int,
) -> None:
    """Print per-case hit-rate, flip-rate, and must-find status across N runs."""
    print(f"\nModel: {model}  Effort: {effort}  Runs: {runs}")
    print("=" * 70)

    n_cases = len(cases)
    total_must_find = 0
    total_must_find_perfect = 0

    for case_idx in range(n_cases):
        case = cases[case_idx]
        run_results = [all_run_results[run_idx][case_idx] for run_idx in range(runs)]
        n_expected = len(case.expected)

        exp_hit_counts: dict[int, int] = defaultdict(int)
        run_hits = []
        run_noise = []

        for r in run_results:
            run_hits.append(r["hits"])
            run_noise.append(r["noise"])
            for exp_idx in r["hit_expected_indices"]:
                exp_hit_counts[exp_idx] += 1

        mean_hits = sum(run_hits) / runs
        mean_noise = sum(run_noise) / runs
        mean_recall = mean_hits / n_expected if n_expected else 0.0

        # Flip-rate: runs where hit count differs from the median (rounded mean)
        majority_hits = round(mean_hits)
        flip_count = sum(1 for h in run_hits if h != majority_hits)
        flip_rate = flip_count / runs

        print(f"\n  [{case.id}]")
        print(
            f"    Recall: {mean_recall:.0%} (mean {mean_hits:.1f}/{n_expected} hits)  "
            f"Noise: {mean_noise:.1f}  Flip-rate: {flip_rate:.0%}"
        )

        for exp_idx, exp in enumerate(case.expected):
            hit_rate = exp_hit_counts[exp_idx] / runs
            must_tag = " [MUST-FIND]" if exp.must_find else ""
            sym = _hit_rate_symbol(hit_rate)
            print(
                f"    {sym} [{hit_rate:.0%} hit-rate] "
                f"line {exp.line_range[0]}-{exp.line_range[1]}  "
                f"{exp.note[:60]}{must_tag}"
            )
            if exp.must_find:
                total_must_find += 1
                if hit_rate >= 1.0:
                    total_must_find_perfect += 1

    # Roll-up
    print("\n" + "=" * 70)
    all_hits_per_run = [
        sum(all_run_results[run_idx][c]["hits"] for c in range(n_cases))
        for run_idx in range(runs)
    ]
    all_noise_per_run = [
        sum(all_run_results[run_idx][c]["noise"] for c in range(n_cases))
        for run_idx in range(runs)
    ]
    all_findings_per_run = [
        sum(all_run_results[run_idx][c]["findings"] for c in range(n_cases))
        for run_idx in range(runs)
    ]
    total_expected_sum = sum(len(c.expected) for c in cases)

    mean_precision = (
        sum(h / f if f else 0.0 for h, f in zip(all_hits_per_run, all_findings_per_run)) / runs
    )
    mean_recall = sum(all_hits_per_run) / (total_expected_sum * runs) if total_expected_sum else 0.0
    mean_noise_rate = (
        sum(n / f if f else 0.0 for n, f in zip(all_noise_per_run, all_findings_per_run)) / runs
    )

    print(
        f"Mean Precision: {mean_precision:.0%}  "
        f"Mean Recall: {mean_recall:.0%}  "
        f"Mean Noise-rate: {mean_noise_rate:.0%}"
    )
    if total_must_find > 0:
        status = "PASS" if total_must_find_perfect == total_must_find else "FAIL"
        print(
            f"Must-find: {total_must_find_perfect}/{total_must_find} "
            f"found in ALL {runs} runs  [{status}]"
        )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "ollama/llama3.2"))
    parser.add_argument("--api-base", default=None,
                        help="Override LLM_API_BASE (e.g. https://openrouter.ai/api/v1)")
    parser.add_argument("--api-key", default=None,
                        help="Override LLM API key (e.g. OPENROUTER_API_KEY value)")
    parser.add_argument("--review-model", default=None,
                        help="Critic/holistic model (SIFT_REVIEW_MODEL). Defaults to --model.")
    parser.add_argument("--review-api-base", default=None,
                        help="API base for the review model. Defaults to --api-base.")
    parser.add_argument("--review-api-key", default=None,
                        help="API key for the review model. Defaults to --api-key.")
    parser.add_argument("--effort", default="balanced", choices=["low", "balanced", "high"])
    parser.add_argument("--case", default=None, help="Run a single case by ID")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run each case (>1 enables consistency reporting)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print findings for each case (single-run mode only)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max parallel cases (lower to avoid rate limits, default 4)",
    )
    args = parser.parse_args()

    config.LLM_MODEL = args.model
    if args.api_base:
        config.LLM_API_BASE = args.api_base.rstrip("/")
    if args.api_key:
        # LiteLLM picks up provider keys from env; set the generic fallback
        os.environ["OPENROUTER_API_KEY"] = args.api_key
        os.environ["ANTHROPIC_API_KEY"] = args.api_key
        os.environ["OPENAI_API_KEY"] = args.api_key
        os.environ["OLLAMA_API_KEY"] = args.api_key
        config.LLM_API_KEY = args.api_key
    if args.review_model:
        config.SIFT_REVIEW_MODEL = args.review_model
    if args.review_api_base:
        config.SIFT_REVIEW_MODEL_BASE_URL = args.review_api_base.rstrip("/")
    elif args.api_base and args.review_model:
        # same base URL for review model unless overridden
        config.SIFT_REVIEW_MODEL_BASE_URL = config.LLM_API_BASE
    if args.review_api_key:
        config.SIFT_REVIEW_MODEL_KEY = args.review_api_key

    plan = plan_for(EffortLevel(args.effort))
    cap = detect(args.model)

    cases = [GoldenCase.load(p) for p in sorted(CASES_DIR.glob("*.json"))]
    if args.case:
        cases = [c for c in cases if c.id == args.case]
        if not cases:
            print(f"No case found with id={args.case!r}")
            sys.exit(1)

    concurrency = min(args.concurrency, len(cases))
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(c):
        async with sem:
            return await score_case(c, plan, cap)

    if args.runs == 1:
        results = list(await asyncio.gather(*[_bounded(c) for c in cases]))
        _print_single_run(results, args.model, args.effort, args.verbose)
    else:
        print(f"Running {args.runs} runs × {len(cases)} case(s)...")
        all_run_results: list[list[dict]] = []
        for run_idx in range(args.runs):
            print(f"  Run {run_idx + 1}/{args.runs}...", end=" ", flush=True)
            run_results = list(
                await asyncio.gather(*[_bounded(c) for c in cases])
            )
            all_run_results.append(run_results)
            hits = sum(r["hits"] for r in run_results)
            expected = sum(r["hits"] + r["misses"] for r in run_results)
            print(f"recall={hits}/{expected}")

        _print_consistency_report(cases, all_run_results, args.model, args.effort, args.runs)


if __name__ == "__main__":
    asyncio.run(main())
