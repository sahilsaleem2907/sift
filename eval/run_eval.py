"""Offline golden-set scorer.

Usage:
    python -m eval.run_eval --model ollama/llama3.2 --effort low
    python -m eval.run_eval --model anthropic/claude-opus-4-8 --effort balanced
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from eval.schema import ExpectedFinding, GoldenCase
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
    """Reconstruct approximate new-file content from a unified diff.

    Uses context lines (' ') and added lines ('+') only, skipping diff headers and
    removed lines. Good enough for tree-sitter function extraction and import graph
    resolution in the eval harness where real file content isn't available.
    """
    lines = []
    for line in file_diff.splitlines():
        if line.startswith(("diff ", "--- ", "+++ ", "@@ ", "index ")):
            continue
        if line.startswith("+"):
            lines.append(line[1:])
        elif line.startswith(" "):
            lines.append(line[1:])
        # skip '-' lines (removed content)
    return "\n".join(lines)


def _build_pr_meta(case: GoldenCase) -> PRMeta:
    file_chunks = split_diff_by_file(case.diff_text)
    # Build pseudo file content from diff +/context lines so tree-sitter can extract
    # function names and the import graph resolver can find import statements.
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
    )


async def score_case(case: GoldenCase, plan, cap) -> dict:
    file_chunks = split_diff_by_file(case.diff_text)
    if not file_chunks:
        file_chunks = [(case.path, case.diff_text)]

    all_findings = []
    for path, file_diff in file_chunks:
        inp = FileReviewInput(
            path=path,
            file_diff=file_diff,
            pr_context={"title": case.description, "body": ""},
        )
        per_file = await run_pipeline_per_file(inp, case.description, plan, cap)
        all_findings.extend(per_file)

    pr_meta = _build_pr_meta(case)
    findings = await run_pipeline_holistic(all_findings, pr_meta, plan, cap)

    hits, misses, noise = 0, 0, 0
    matched_finding_ids: set[int] = set()
    for exp in case.expected:
        matched = [f for f in findings if _is_hit(f, exp)]
        if matched:
            hits += 1
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
        "_findings": findings,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL", "ollama/llama3.2"),
    )
    parser.add_argument(
        "--effort",
        default="balanced",
        choices=["low", "balanced", "high"],
    )
    parser.add_argument("--case", default=None, help="Run a single case by ID")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print findings for each case")
    args = parser.parse_args()

    config.LLM_MODEL = args.model
    plan = plan_for(EffortLevel(args.effort))
    cap = detect(args.model)

    cases = [GoldenCase.load(p) for p in sorted(CASES_DIR.glob("*.json"))]
    if args.case:
        cases = [c for c in cases if c.id == args.case]
        if not cases:
            print(f"No case found with id={args.case!r}")
            sys.exit(1)

    results = await asyncio.gather(*[score_case(c, plan, cap) for c in cases])

    total_expected = sum(r["hits"] + r["misses"] for r in results)
    total_hits = sum(r["hits"] for r in results)
    total_noise = sum(r["noise"] for r in results)
    total_findings = sum(r["findings"] for r in results)

    precision = total_hits / total_findings if total_findings else 0.0
    recall = total_hits / total_expected if total_expected else 0.0
    noise_rate = total_noise / total_findings if total_findings else 0.0

    print(f"\nModel: {args.model}  Effort: {args.effort}")
    print(
        f"Precision: {precision:.0%}  Recall: {recall:.0%}  "
        f"Noise-rate: {noise_rate:.0%}"
    )
    print(
        f"  ({total_hits}/{total_expected} expected hits, "
        f"{total_noise} noise findings)\n"
    )
    for r in results:
        status = "OK" if r["misses"] == 0 else "MISS"
        print(
            f"  [{status}] {r['case']}: hits={r['hits']} "
            f"misses={r['misses']} noise={r['noise']}"
        )
        if args.verbose and r["_findings"]:
            for f in r["_findings"]:
                print(
                    f"         path={f.path} line={f.line} "
                    f"impact={f.impact.value} certainty={f.certainty.value} "
                    f"category={f.category} origin={f.origin}"
                )


if __name__ == "__main__":
    asyncio.run(main())
