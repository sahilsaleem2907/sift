"""Orchestrate PR review: diff -> split by file -> per-file LLM -> summary -> post comments + issue comment -> store."""
import hashlib
import logging
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from src.integrations.github_client import GitHubClient, get_installation_token
from src.core.pr_analyzer import get_diff_for_review, get_diff_line_numbers, split_diff_by_file
from src.core.semgrep_runner import run_semgrep
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
            findings_by_path: Dict[str, List[dict]] = run_semgrep(path_to_content)

            # Group by diff content so we don't run the LLM for the same code block multiple times
            diff_to_paths: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
            for path, file_diff in file_chunks:
                if not file_diff.strip():
                    continue
                diff_to_paths[_diff_content_key(file_diff)].append((path, file_diff))

            collected: List[Dict[str, Any]] = []
            for _content_key, path_diff_list in diff_to_paths.items():
                path0, file_diff = path_diff_list[0]
                diff_lines = diff_lines_per_path.get(path0, set())
                findings_on_diff = [
                    f
                    for f in findings_by_path.get(path0, [])
                    if f.get("line") in diff_lines
                ]
                file_pr_context: Dict[str, Any] = {**(pr_context or {}), "semgrep_findings": findings_on_diff}
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
