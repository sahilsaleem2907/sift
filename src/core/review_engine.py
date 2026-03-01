"""Orchestrate PR review: diff -> split by file -> per-file LLM -> summary -> post comments + issue comment -> store."""
import hashlib
import logging
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from src import config
from src.integrations.github_client import GitHubClient, get_installation_token
from src.core.pr_analyzer import get_diff_for_review, get_diff_line_numbers, split_diff_by_file
from src.core.linter_runner import run_linters
from src.core.semgrep_runner import run_semgrep
from src.core.repo_cache import get_repo_at_commit
from src.core.codeql_runner import run_codeql, languages_from_paths
from src.core.analysis_routing import (
    FileType,
    classify_file_type,
    get_tools_for_file,
    risk_level,
    score_risk_with_breakdown,
)
# from src.intelligence.ast.diff_ast import build_diff_ast
from src.intelligence.llm_client import review_file, summarize_review
from src.storage.database import store_review

logger = logging.getLogger(__name__)


def _diff_content_key(file_diff: str) -> str:
    """Stable hash for diff content so we don't run the LLM for the same code block twice."""
    return hashlib.sha256(file_diff.strip().encode("utf-8")).hexdigest()


def _merge_comments_by_line(collected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One comment per (path, line). If multiple bodies for same line, merge with bullet points."""
    by_key: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for item in collected:
        key = (item["path"], item["line"])
        body = (item.get("body") or "").strip()
        if body:
            by_key[key].append(body)
    merged: List[Dict[str, Any]] = []
    for (path, line), bodies in by_key.items():
        if len(bodies) == 1:
            merged.append({"path": path, "line": line, "body": bodies[0]})
        else:
            merged.append(
                {
                    "path": path,
                    "line": line,
                    "body": "**Issues:**\n" + "\n".join(f"- {b}" for b in bodies),
                }
            )
    return merged


async def run_review(owner: str, repo: str, pr_number: int, installation_id: int) -> None:
    """Run the full review flow: fetch diff, split by file, per-file LLM, summarize, post inline comments + summary, store.

    Logs and swallows exceptions so the webhook response is not affected.
    """
    repo_full = f"{owner}/{repo}"
    try:
        token = await get_installation_token(installation_id)
        async with GitHubClient(installation_id, token=token) as github:
            logger.info("Starting review for %s PR #%s", repo_full, pr_number)

            diff, pr_context = await get_diff_for_review(owner, repo, pr_number, github)
            if not diff.strip():
                logger.warning("Empty diff for %s PR #%s", repo_full, pr_number)
                return

            commit_id = await github.get_pr_head_commit(owner, repo, pr_number)
            file_chunks = split_diff_by_file(diff)
            if not file_chunks:
                logger.warning("No file chunks from diff for %s PR #%s", repo_full, pr_number)
                return

            diff_lines_per_path: Dict[str, Set[int]] = {
                path: get_diff_line_numbers(fd) for path, fd in file_chunks
            }

            path_to_content: Dict[str, str] = {}
            for path, _ in file_chunks:
                content = await github.get_file_content(owner, repo, path, commit_id)
                if content is not None:
                    path_to_content[path] = content

            # Smart routing: classify and score each path; build tool path sets
            path_to_file_type: Dict[str, FileType] = {}
            path_to_risk: Dict[str, Any] = {}  # RiskLevel
            linter_paths: Set[str] = set()
            semgrep_paths: Set[str] = set()
            codeql_paths: Set[str] = set()

            if config.SIFT_SMART_ROUTING_ENABLED:
                pr_paths = [p for p, _ in file_chunks]
                logger.debug(
                    "[Smart routing] Classifying %d file(s) and computing risk/tools",
                    len(pr_paths),
                )
                for path in pr_paths:
                    ft = classify_file_type(path)
                    content = path_to_content.get(path) or ""
                    sc, breakdown = score_risk_with_breakdown(path, content, ft)
                    rl = risk_level(sc)
                    path_to_file_type[path] = ft
                    path_to_risk[path] = rl
                    tools_set = get_tools_for_file(ft, rl)
                    if "linter" in tools_set:
                        linter_paths.add(path)
                    if "semgrep" in tools_set:
                        semgrep_paths.add(path)
                    if "codeql" in tools_set:
                        codeql_paths.add(path)
                    tools_str = ",".join(sorted(tools_set)) if tools_set else "SKIP"
                    # Log why this risk level: which factors contributed
                    parts = [f"{k}+{v}" for k, v in breakdown.items() if v > 0]
                    reason = " ".join(parts) if parts else "no factors"
                    logger.debug(
                        "[Smart routing] %s → type=%s score=%s level=%s → tools=[%s]",
                        path, ft.value, sc, rl.value, tools_str,
                    )
                    logger.debug(
                        "[Smart routing]   risk reason: %s (total=%s)",
                        reason, sc,
                    )
                skip_count = sum(
                    1 for p in pr_paths
                    if path_to_file_type.get(p) in (FileType.DOCUMENTATION, FileType.ASSETS)
                )
                logger.debug(
                    "[Smart routing] Summary: linter=%d paths, semgrep=%d paths, codeql=%d paths, skip(docs/assets)=%d",
                    len(linter_paths),
                    len(semgrep_paths),
                    len(codeql_paths),
                    skip_count,
                )

            if config.SIFT_SMART_ROUTING_ENABLED:
                linter_input = {p: path_to_content[p] for p in linter_paths if p in path_to_content}
                semgrep_input = {p: path_to_content[p] for p in semgrep_paths if p in path_to_content}
                logger.debug(
                    "[Smart routing] Running linter on %d path(s): %s",
                    len(linter_input), sorted(linter_input.keys()) if linter_input else [],
                )
                logger.debug(
                    "[Smart routing] Running Semgrep on %d path(s): %s",
                    len(semgrep_input), sorted(semgrep_input.keys()) if semgrep_input else [],
                )
                findings_by_path = run_semgrep(semgrep_input)
                linter_issues_by_path = run_linters(linter_input)
            else:
                findings_by_path = run_semgrep(path_to_content)
                linter_issues_by_path = run_linters(path_to_content)

            codeql_findings_by_path: Dict[str, List[dict]] = {}
            run_codeql_this_pr = config.CODEQL_ENABLED and (
                not config.SIFT_SMART_ROUTING_ENABLED or len(codeql_paths) > 0
            )
            if config.SIFT_SMART_ROUTING_ENABLED and config.CODEQL_ENABLED:
                if codeql_paths:
                    logger.debug(
                        "[Smart routing] Running CodeQL (CRITICAL code paths): %s",
                        sorted(codeql_paths),
                    )
                else:
                    logger.debug(
                        "[Smart routing] Skipping CodeQL (no CRITICAL code files in this PR)",
                    )
            if run_codeql_this_pr:
                try:
                    source_root = get_repo_at_commit(owner, repo, commit_id, token)
                    codeql_langs = languages_from_paths([p for p, _ in file_chunks])
                    codeql_findings_by_path = run_codeql(
                        source_root,
                        config.CODEQL_SUITE,
                        codeql_langs,
                        config.CODEQL_TIMEOUT,
                    )
                    if codeql_findings_by_path:
                        total_codeql = sum(len(v) for v in codeql_findings_by_path.values())
                        logger.debug(
                            "CodeQL (entire repo): %d path(s), %d total findings: %s",
                            len(codeql_findings_by_path),
                            total_codeql,
                            {p: len(findings) for p, findings in codeql_findings_by_path.items()},
                        )
                        logger.debug("CodeQL findings for entire repo: %s", codeql_findings_by_path)
                except Exception as e:
                    logger.warning("CodeQL skipped: %s", e)
            total_linter_issues = sum(len(v) for v in linter_issues_by_path.values())
            if linter_issues_by_path:
                logger.debug(
                    "Linters completed: %d path(s) with issues, %d total issues: %s",
                    len(linter_issues_by_path),
                    total_linter_issues,
                    {p: len(issues) for p, issues in linter_issues_by_path.items()},
                )

            # Group by diff content so we don't run the LLM for the same code block multiple times
            diff_to_paths: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
            for path, file_diff in file_chunks:
                if not file_diff.strip():
                    continue
                diff_to_paths[_diff_content_key(file_diff)].append((path, file_diff))

            collected: List[Dict[str, Any]] = []
            for _content_key, path_diff_list in diff_to_paths.items():
                path0, file_diff = path_diff_list[0]

                # Skip docs/assets when smart routing is enabled
                if config.SIFT_SMART_ROUTING_ENABLED:
                    ft0 = path_to_file_type.get(path0)
                    if ft0 in (FileType.DOCUMENTATION, FileType.ASSETS):
                        logger.debug(
                            "[Smart routing] Skipping LLM review (docs/assets): %s",
                            path0,
                        )
                        continue

                diff_lines = diff_lines_per_path.get(path0, set())
                # Only attach semgrep/codeql findings when this path was in the tool set
                if config.SIFT_SMART_ROUTING_ENABLED and path0 not in semgrep_paths:
                    findings_on_diff = []
                else:
                    findings_on_diff = [
                        f
                        for f in findings_by_path.get(path0, [])
                        if f.get("line") in diff_lines
                    ]
                if config.SIFT_SMART_ROUTING_ENABLED and path0 not in codeql_paths:
                    codeql_on_diff = []
                else:
                    codeql_on_diff = [
                        f
                        for f in codeql_findings_by_path.get(path0, [])
                        if f.get("line") in diff_lines
                    ]
                if config.SIFT_SMART_ROUTING_ENABLED:
                    logger.debug(
                        "[Smart routing] LLM context for %s: linter=%s semgrep=%s codeql=%s (on-diff lines)",
                        path0,
                        "yes" if path0 in linter_paths else "no",
                        len(findings_on_diff),
                        len(codeql_on_diff),
                    )
                file_pr_context: Dict[str, Any] = {
                    **(pr_context or {}),
                    "semgrep_findings": findings_on_diff,
                    "codeql_findings": codeql_on_diff,
                }

                # source = path_to_content.get(path0)
                # if source is not None:
                #     try:
                #         ast_diff = build_diff_ast(path0, source, file_diff)
                #         if ast_diff is not None:
                #             file_pr_context["ast_diff"] = ast_diff
                #     except Exception as e:
                #         logger.warning("build_diff_ast failed for %s: %s", path0, e)
                raw_linter_list = linter_issues_by_path.get(path0, []) if (not config.SIFT_SMART_ROUTING_ENABLED or path0 in linter_paths) else []
                raw_linter_count = len(raw_linter_list)
                linter_on_diff = [
                    i for i in raw_linter_list
                    if i.get("line") in diff_lines
                ]
                if raw_linter_count > 0:
                    logger.debug(
                        "Linter filter: path=%s, raw_issues=%d, on_diff_lines=%d, diff_line_set_size=%d",
                        path0,
                        raw_linter_count,
                        len(linter_on_diff),
                        len(diff_lines),
                    )
                file_lines = (path_to_content.get(path0) or "").splitlines()
                linter_issues_with_snippets: List[Dict[str, Any]] = []
                for i in linter_on_diff:
                    line_no = i.get("line")
                    snippet = ""
                    if line_no is not None and 1 <= line_no <= len(file_lines):
                        snippet = file_lines[line_no - 1].strip()
                    linter_issues_with_snippets.append({
                        **i,
                        "snippet": snippet,
                    })
                file_pr_context = {
                    **(pr_context or {}),
                    "semgrep_findings": findings_on_diff,
                    "codeql_findings": codeql_on_diff,
                    "linter_issues": linter_issues_with_snippets,
                }
                try:
                    comments = await review_file(file_diff, path0, file_pr_context)
                    # One comment per line for this code block (LLM might return duplicate lines)
                    seen_line: set[int] = set()
                    for c in comments:
                        if c["line"] in seen_line:
                            continue
                        seen_line.add(c["line"])
                        for path, _ in path_diff_list:
                            collected.append(
                                {"path": path, "line": c["line"], "body": c["body"]}
                            )
                except Exception as e:
                    logger.warning("review_file failed for %s: %s", path0, e)

            # One comment per (path, line); merge multiple into bullet points
            collected = _merge_comments_by_line(collected)

            collected = [
                c
                for c in collected
                if c["line"] in diff_lines_per_path.get(c["path"], set())
            ]
            summary = await summarize_review(collected) if collected else "No inline comments for this review."
            if not summary.strip():
                summary = "Review completed with inline comments on the Files changed tab."

            # Post all inline review comments first (Files changed tab), then the summary (Conversation tab)
            for item in collected:
                try:
                    await github.create_review_comment(
                        owner,
                        repo,
                        pr_number,
                        commit_id=commit_id,
                        path=item["path"],
                        line=item["line"],
                        body=item["body"],
                        side="RIGHT",
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to post review comment %s:%s: %s",
                        item.get("path"),
                        item.get("line"),
                        e,
                    )

            # Summary comment must happen after all review comments
            comment_id = await github.create_comment(owner, repo, pr_number, summary)
            try:
                store_review(
                    repo_full,
                    pr_number,
                    installation_id,
                    summary,
                    comment_id=comment_id,
                )
            except Exception as e:
                logger.warning("Failed to store review in DB: %s", e)
            logger.info("Review completed for %s PR #%s", repo_full, pr_number)
    except Exception as e:
        logger.exception("Review failed for %s PR #%s: %s", repo_full, pr_number, e)
