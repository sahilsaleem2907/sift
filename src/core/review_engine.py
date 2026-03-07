"""Orchestrate PR review: diff -> split by file -> per-file LLM -> summary -> post comments + issue comment -> store."""
import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src import config
from src.integrations.github_client import GitHubClient, get_installation_token
from src.core.pr_analyzer import get_diff_for_review, get_diff_line_numbers, split_diff_by_file
from src.core.linter_runner import run_linters, _detect_linter
from src.core.semgrep_runner import run_semgrep
from src.core.repo_cache import get_repo_at_commit
from src.core.codeql_runner import run_codeql, languages_from_paths
from src.core.analysis_routing import (
    FileType,
    classify_file_type,
    get_tools_for_file,
    risk_level,
    score_risk_combined,
)
# from src.intelligence.ast.diff_ast import build_diff_ast
from src.intelligence.ast.function_extract import extract_modified_functions
from src.intelligence.llm_client import review_file, summarize_review
from src.storage.database import (
    get_avg_quality_score_for_path_pattern,
    get_tool_cache_hits,
    store_review,
    store_tool_cache,
)

logger = logging.getLogger(__name__)


# Security-sensitive function name substrings for AST-based risk boost
_AST_FUNCTION_RISK_KEYWORDS = frozenset({
    "auth", "verify", "validate", "encrypt", "decrypt", "login",
    "check_permission", "sanitize", "hash",
})
_AST_FUNCTION_BOOST = 15


def _has_security_sensitive_function(mod_funcs: list) -> bool:
    """True if any modified function name contains security-sensitive keywords."""
    for chunk in mod_funcs:
        name = (chunk.name or "").lower()
        for kw in _AST_FUNCTION_RISK_KEYWORDS:
            if kw in name:
                return True
    return False


def _diff_content_key(file_diff: str) -> str:
    """Stable hash for diff content so we don't run the LLM for the same code block twice."""
    return hashlib.sha256(file_diff.strip().encode("utf-8")).hexdigest()


def _tool_cache_key(tool: str, path: str, content: str, linter_name: Optional[str] = None) -> Optional[str]:
    """Cache key for tool result. For linter, linter_name required (from _detect_linter). Returns None if not cacheable."""
    if tool == "semgrep":
        return hashlib.sha256(("semgrep" + content).encode("utf-8")).hexdigest()
    if tool == "linter":
        if not linter_name:
            return None
        return hashlib.sha256(("linter:" + linter_name + ":" + content).encode("utf-8")).hexdigest()
    return None


def _check_and_split_cache(
    tool: str,
    path_to_content: Dict[str, str],
    ttl_hours: int,
) -> Tuple[Dict[str, List[Any]], Dict[str, str]]:
    """Return (cached_results_by_path, uncached_input). uncached_input is path -> content for cache misses."""
    if not path_to_content or ttl_hours <= 0:
        return {}, dict(path_to_content)
    path_to_key: Dict[str, str] = {}
    for path, content in path_to_content.items():
        linter_name = _detect_linter(path) if tool == "linter" else None
        key = _tool_cache_key(tool, path, content, linter_name)
        if key is not None:
            path_to_key[path] = key
    if not path_to_key:
        return {}, dict(path_to_content)
    keys = list(path_to_key.values())
    hits = get_tool_cache_hits(keys, ttl_hours)
    cached_results: Dict[str, List[Any]] = {
        path: hits[key] for path, key in path_to_key.items() if key in hits
    }
    uncached_input = {
        path: content for path, content in path_to_content.items()
        if path_to_key.get(path) not in hits
    }
    return cached_results, uncached_input


def _store_results_cache(
    tool: str,
    path_to_content: Dict[str, str],
    results_by_path: Dict[str, List[Any]],
) -> None:
    """Store tool results in cache. path_to_content is the input that was run (e.g. uncached subset)."""
    if not path_to_content:
        return
    entries: List[Dict[str, Any]] = []
    for path, content in path_to_content.items():
        linter_name = _detect_linter(path) if tool == "linter" else None
        key = _tool_cache_key(tool, path, content, linter_name)
        if key is None:
            continue
        findings = results_by_path.get(path, [])
        entries.append({
            "cache_key": key,
            "tool": tool,
            "findings_json": json.dumps(findings),
        })
    if entries:
        store_tool_cache(entries)


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
                # Build feedback cache: path_prefix -> avg quality (one query per unique dir)
                path_prefix_to_quality: Dict[str, Optional[float]] = {}
                for path in pr_paths:
                    parts = path.replace("\\", "/").split("/")
                    prefix = "/".join(parts[:-1]) if len(parts) > 1 else ""
                    if prefix and prefix not in path_prefix_to_quality:
                        path_prefix_to_quality[prefix] = get_avg_quality_score_for_path_pattern(
                            repo_full, prefix
                        )
                for path, file_diff in file_chunks:
                    ft = classify_file_type(path)
                    content = path_to_content.get(path) or ""
                    sc, breakdown = score_risk_combined(path, content, ft, file_diff)
                    # Feedback loop: historical quality for this path's directory
                    parts = path.replace("\\", "/").split("/")
                    path_prefix = "/".join(parts[:-1]) if len(parts) > 1 else ""
                    avg_quality = path_prefix_to_quality.get(path_prefix) if path_prefix else None
                    if avg_quality is not None:
                        if avg_quality < 35:
                            sc += 10
                            breakdown["feedback"] = 10
                        elif avg_quality > 75:
                            sc -= 5
                            breakdown["feedback"] = -5
                    # AST-based boost: security-sensitive function names
                    if ft == FileType.CODE:
                        try:
                            mod_funcs = extract_modified_functions(path, content, file_diff)
                            if mod_funcs and _has_security_sensitive_function(mod_funcs):
                                sc += _AST_FUNCTION_BOOST
                                breakdown["ast_function"] = _AST_FUNCTION_BOOST
                        except Exception as e:
                            logger.debug("AST function extract failed for %s: %s", path, e)
                    rl = risk_level(sc)
                    path_to_file_type[path] = ft
                    path_to_risk[path] = rl
                    tools_set = get_tools_for_file(ft, rl, path)
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
            else:
                semgrep_input = path_to_content
                linter_input = path_to_content

            # Tool result cache: split into cached vs uncached so we only run on misses
            ttl_hours = config.TOOL_CACHE_TTL_HOURS if config.TOOL_CACHE_ENABLED else 0
            if config.TOOL_CACHE_ENABLED and ttl_hours > 0:
                semgrep_cached, semgrep_uncached = _check_and_split_cache(
                    "semgrep", semgrep_input, ttl_hours
                )
                linter_cached, linter_uncached = _check_and_split_cache(
                    "linter", linter_input, ttl_hours
                )
                if semgrep_cached:
                    logger.debug(
                        "[Tool cache REUSED] Semgrep: %d path(s) skipped run (using cached results): %s",
                        len(semgrep_cached),
                        sorted(semgrep_cached.keys()),
                    )
                if linter_cached:
                    logger.debug(
                        "[Tool cache REUSED] Linter: %d path(s) skipped run (using cached results): %s",
                        len(linter_cached),
                        sorted(linter_cached.keys()),
                    )
            else:
                semgrep_cached, semgrep_uncached = {}, semgrep_input
                linter_cached, linter_uncached = {}, linter_input

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

            codeql_cache_key: Optional[str] = None
            if run_codeql_this_pr and config.TOOL_CACHE_ENABLED and ttl_hours > 0:
                codeql_langs_for_key = languages_from_paths([p for p, _ in file_chunks])
                codeql_cache_key = hashlib.sha256(
                    (
                        "codeql:" + repo_full + ":" + commit_id + ":"
                        + config.CODEQL_SUITE + ":" + ",".join(sorted(codeql_langs_for_key))
                    ).encode("utf-8")
                ).hexdigest()

            async def _run_codeql_task() -> Dict[str, List[dict]]:
                def _run() -> Dict[str, List[dict]]:
                    if codeql_cache_key and config.TOOL_CACHE_ENABLED and ttl_hours > 0:
                        hits = get_tool_cache_hits([codeql_cache_key], ttl_hours)
                        if codeql_cache_key in hits:
                            cached = hits[codeql_cache_key]
                            if isinstance(cached, dict):
                                logger.debug(
                                    "[Tool cache REUSED] CodeQL: using cached results for %s (skipped run)",
                                    repo_full,
                                )
                                return cached
                    try:
                        source_root = get_repo_at_commit(owner, repo, commit_id, token)
                        codeql_langs = languages_from_paths([p for p, _ in file_chunks])
                        result = run_codeql(
                            source_root,
                            config.CODEQL_SUITE,
                            codeql_langs,
                            config.CODEQL_TIMEOUT,
                        )
                        if codeql_cache_key and config.TOOL_CACHE_ENABLED and result:
                            store_tool_cache([{
                                "cache_key": codeql_cache_key,
                                "tool": "codeql",
                                "findings_json": json.dumps(result),
                            }])
                        return result
                    except Exception as e:
                        logger.warning("CodeQL skipped: %s", e)
                        return {}

                return await asyncio.to_thread(_run)

            async def _codeql_or_empty() -> Dict[str, List[dict]]:
                if not run_codeql_this_pr:
                    return {}
                return await _run_codeql_task()

            semgrep_result, linter_result, codeql_result = await asyncio.gather(
                asyncio.to_thread(run_semgrep, semgrep_uncached),
                asyncio.to_thread(run_linters, linter_uncached),
                _codeql_or_empty(),
                return_exceptions=True,
            )

            if isinstance(semgrep_result, BaseException):
                logger.warning("Semgrep failed: %s", semgrep_result)
                findings_by_path = dict(semgrep_cached)
                if semgrep_cached:
                    logger.debug(
                        "[Tool cache REUSED] Semgrep: using %d cached path(s) only (run failed)",
                        len(semgrep_cached),
                    )
            else:
                findings_by_path = {**semgrep_cached, **semgrep_result}
                if semgrep_cached or semgrep_result:
                    logger.debug(
                        "[Tool cache REUSED] Semgrep: %d from cache, %d from run, total %d path(s)",
                        len(semgrep_cached),
                        len(semgrep_result),
                        len(findings_by_path),
                    )
                if config.TOOL_CACHE_ENABLED and semgrep_uncached:
                    _store_results_cache("semgrep", semgrep_uncached, semgrep_result)
            if isinstance(linter_result, BaseException):
                logger.warning("Linters failed: %s", linter_result)
                linter_issues_by_path = dict(linter_cached)
                if linter_cached:
                    logger.debug(
                        "[Tool cache REUSED] Linter: using %d cached path(s) only (run failed)",
                        len(linter_cached),
                    )
            else:
                linter_issues_by_path = {**linter_cached, **linter_result}
                if linter_cached or linter_result:
                    logger.debug(
                        "[Tool cache REUSED] Linter: %d from cache, %d from run, total %d path(s)",
                        len(linter_cached),
                        len(linter_result),
                        len(linter_issues_by_path),
                    )
                if config.TOOL_CACHE_ENABLED and linter_uncached:
                    _store_results_cache("linter", linter_uncached, linter_result)
            if isinstance(codeql_result, BaseException):
                logger.warning("CodeQL failed: %s", codeql_result)
                codeql_findings_by_path = {}
            else:
                codeql_findings_by_path = codeql_result

            if findings_by_path:
                total_semgrep = sum(len(v) for v in findings_by_path.values())
                logger.debug(
                    "Semgrep (entire run): %d path(s), %d total findings: %s",
                    len(findings_by_path),
                    total_semgrep,
                    {p: len(f) for p, f in findings_by_path.items()},
                )
                logger.debug("Semgrep findings fully: %s", findings_by_path)
            if codeql_findings_by_path:
                total_codeql = sum(len(v) for v in codeql_findings_by_path.values())
                logger.debug(
                    "CodeQL (entire repo): %d path(s), %d total findings: %s",
                    len(codeql_findings_by_path),
                    total_codeql,
                    {p: len(findings) for p, findings in codeql_findings_by_path.items()},
                )
                logger.debug("CodeQL findings for entire repo: %s", codeql_findings_by_path)
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

            _vector_upsert_queue: List[Tuple[list, list]] = []
            if config.VECTOR_DB_ENABLED:
                logger.debug(
                    "[Vector] feature enabled for this review (repo=%s): will extract modified functions, search similar, and upsert chunks",
                    repo_full,
                )

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
                # Semgrep: diff-filtered + critical (ERROR severity) bypass
                if config.SIFT_SMART_ROUTING_ENABLED and path0 not in semgrep_paths:
                    semgrep_for_llm: List[Dict[str, Any]] = []
                else:
                    all_semgrep = findings_by_path.get(path0, [])
                    semgrep_on_diff = [f for f in all_semgrep if f.get("line") in diff_lines]
                    semgrep_critical = [
                        {**f, "critical_bypass": True}
                        for f in all_semgrep
                        if f not in semgrep_on_diff and (f.get("severity") or "").upper() == "ERROR"
                    ]
                    semgrep_for_llm = semgrep_on_diff + semgrep_critical
                    if all_semgrep:
                        logger.debug(
                            "Semgrep filter path=%s: raw=%d, on_diff=%d, critical_bypass=%d, for_llm=%d",
                            path0,
                            len(all_semgrep),
                            len(semgrep_on_diff),
                            len(semgrep_critical),
                            len(semgrep_for_llm),
                        )
                        logger.debug(
                            "Semgrep findings fully path=%s: all=%s on_diff=%s critical=%s for_llm=%s",
                            path0,
                            all_semgrep,
                            semgrep_on_diff,
                            semgrep_critical,
                            semgrep_for_llm,
                        )
                # CodeQL: same pattern
                if config.SIFT_SMART_ROUTING_ENABLED and path0 not in codeql_paths:
                    codeql_for_llm: List[Dict[str, Any]] = []
                else:
                    all_codeql = codeql_findings_by_path.get(path0, [])
                    codeql_on_diff = [f for f in all_codeql if f.get("line") in diff_lines]
                    codeql_critical = [
                        {**f, "critical_bypass": True}
                        for f in all_codeql
                        if f not in codeql_on_diff and (f.get("severity") or "").upper() == "ERROR"
                    ]
                    codeql_for_llm = codeql_on_diff + codeql_critical
                if config.SIFT_SMART_ROUTING_ENABLED:
                    logger.debug(
                        "[Smart routing] LLM context for %s: linter=%s semgrep=%s codeql=%s",
                        path0,
                        "yes" if path0 in linter_paths else "no",
                        len(semgrep_for_llm),
                        len(codeql_for_llm),
                    )
                file_pr_context: Dict[str, Any] = {
                    **(pr_context or {}),
                    "semgrep_findings": semgrep_for_llm,
                    "codeql_findings": codeql_for_llm,
                }

                raw_linter_list = linter_issues_by_path.get(path0, []) if (not config.SIFT_SMART_ROUTING_ENABLED or path0 in linter_paths) else []
                raw_linter_count = len(raw_linter_list)
                linter_on_diff = [i for i in raw_linter_list if i.get("line") in diff_lines]
                linter_critical = [
                    {**i, "critical_bypass": True}
                    for i in raw_linter_list
                    if i not in linter_on_diff and (i.get("severity") or "").lower() == "error"
                ]
                linter_for_llm = linter_on_diff + linter_critical
                if raw_linter_count > 0:
                    logger.debug(
                        "Linter filter: path=%s, raw=%d, on_diff=%d, critical_bypass=%d",
                        path0,
                        raw_linter_count,
                        len(linter_on_diff),
                        len(linter_critical),
                    )
                file_lines = (path_to_content.get(path0) or "").splitlines()
                linter_issues_with_snippets: List[Dict[str, Any]] = []
                for i in linter_for_llm:
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
                    "semgrep_findings": semgrep_for_llm,
                    "codeql_findings": codeql_for_llm,
                    "linter_issues": linter_issues_with_snippets,
                }

                if config.VECTOR_DB_ENABLED:
                    try:
                        from src.intelligence.embeddings import get_embeddings
                        from src.storage.vector_store import search_similar, upsert_chunks

                        logger.debug("[Vector] path=%s: extracting modified functions from diff", path0)
                        mod_funcs = extract_modified_functions(
                            path0, path_to_content.get(path0, ""), file_diff
                        )
                        if not mod_funcs:
                            logger.debug("[Vector] path=%s: no modified functions in diff, skipping embed/search", path0)
                        else:
                            logger.debug(
                                "[Vector] path=%s: extracted %d function(s), hashes=%s",
                                path0, len(mod_funcs), [f.content_hash[:12] + "..." for f in mod_funcs],
                            )
                            func_embeddings = await get_embeddings([f.text for f in mod_funcs])
                            logger.debug("[Vector] path=%s: embedded %d function(s)", path0, len(func_embeddings))
                            all_matches = []
                            exclude_hashes = {f.content_hash for f in mod_funcs}
                            exclude_path = path0 if config.VECTOR_EXCLUDE_SAME_FILE else None
                            logger.debug(
                                "[Vector] path=%s: search params exclude_hashes=%d, exclude_path=%s, top_k=%s",
                                path0, len(exclude_hashes), exclude_path, config.VECTOR_SIMILARITY_TOP_K,
                            )
                            for idx, emb in enumerate(func_embeddings):
                                matches = search_similar(
                                    repo_full, emb, exclude_hashes, exclude_path,
                                    config.VECTOR_SIMILARITY_TOP_K,
                                )
                                logger.debug(
                                    "[Vector] path=%s: query %d/%d returned %d similar chunk(s)",
                                    path0, idx + 1, len(func_embeddings), len(matches),
                                )
                                all_matches.extend(matches)
                            seen_hashes: Dict[str, Any] = {}
                            for m in all_matches:
                                if m.content_hash not in seen_hashes or m.score > seen_hashes[m.content_hash].score:
                                    seen_hashes[m.content_hash] = m
                            unique_matches = sorted(
                                seen_hashes.values(), key=lambda m: m.score, reverse=True
                            )[:config.VECTOR_SIMILARITY_TOP_K]
                            logger.debug(
                                "[Vector] path=%s: after dedupe %d unique match(es), top scores=%s",
                                path0, len(unique_matches),
                                [round(m.score, 4) for m in unique_matches[:5]] if unique_matches else [],
                            )
                            if unique_matches:
                                file_pr_context["similar_snippets"] = unique_matches
                                logger.debug(
                                    "[Vector] path=%s: injected similar_snippets (%d) into LLM context",
                                    path0, len(unique_matches),
                                )
                            _vector_upsert_queue.append((mod_funcs, func_embeddings))
                            logger.debug("[Vector] path=%s: queued %d chunk(s) for upsert", path0, len(mod_funcs))
                    except Exception as e:
                        logger.warning("Vector similarity failed for %s: %s", path0, e)

                try:
                    comments = await review_file(file_diff, path0, file_pr_context)
                    for c in comments:
                        for path, _ in path_diff_list:
                            collected.append(
                                {"path": path, "line": c["line"], "body": c["body"]}
                            )
                except Exception as e:
                    logger.warning("review_file failed for %s: %s", path0, e)

            if config.VECTOR_DB_ENABLED and _vector_upsert_queue:
                try:
                    from src.storage.vector_store import upsert_chunks
                    logger.debug(
                        "[Vector] repo=%s: starting batch upsert of %d file batch(es), total chunks=%s",
                        repo_full, len(_vector_upsert_queue),
                        sum(len(f) for f, _ in _vector_upsert_queue),
                    )
                    for batch_idx, (_funcs, _embs) in enumerate(_vector_upsert_queue):
                        upsert_chunks(repo_full, _funcs, _embs)
                        logger.debug(
                            "[Vector] repo=%s: upsert batch %d/%d done (%d chunks)",
                            repo_full, batch_idx + 1, len(_vector_upsert_queue), len(_funcs),
                        )
                    logger.debug(
                        "[Vector] repo=%s: completed all upserts (%d batch(es))",
                        repo_full, len(_vector_upsert_queue),
                    )
                except Exception as e:
                    logger.warning("Vector upsert failed for %s: %s", repo_full, e)

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
                    paths=[p for p, _ in file_chunks],
                )
            except Exception as e:
                logger.warning("Failed to store review in DB: %s", e)
            logger.info("Review completed for %s PR #%s", repo_full, pr_number)
    except Exception as e:
        logger.exception("Review failed for %s PR #%s: %s", repo_full, pr_number, e)
