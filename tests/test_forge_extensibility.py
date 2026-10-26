"""Prove that a non-GitHub forge can be registered and drive run_review end-to-end.

No real HTTP calls, no DB, no LLM — all external boundaries are faked in-memory.
This is the core proof that the open-core seam works: a second provider drives
the entire review pipeline without changing any file in sift/core/.
"""
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sift.integrations.base import ForgeProvider
from sift.integrations.registry import get_forge, register_forge


FAKE_DIFF = """\
diff --git a/app/calc.py b/app/calc.py
--- a/app/calc.py
+++ b/app/calc.py
@@ -1,2 +1,3 @@
 x = 1
+y = None
+print(y.upper())
"""

FAKE_PR_DETAILS = {
    "title": "Add calc feature",
    "body": "Adds a calculation",
    "head_sha": "abc123",
}

FAKE_COMMENT_BODY = (
    "![BUG](https://img.shields.io/badge/BUG-AA0000?style=for-the-badge) Null dereference\n\n"
    "`y` may be None before `.upper()` is called."
)


async def _critic_passthrough(findings, *args, **kwargs):
    return findings


class DummyForge(ForgeProvider):
    """In-memory forge implementation for tests. Records all write calls."""

    def __init__(self) -> None:
        self.posted_reviews: List[Dict[str, Any]] = []
        self.posted_comments: List[str] = []
        self.set_statuses: List[tuple] = []

    async def __aenter__(self) -> "DummyForge":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def get_clone_token(self) -> Optional[str]:
        return None

    async def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        return FAKE_DIFF

    async def get_compare_diff(
        self, owner: str, repo: str, base_sha: str, head_sha: str
    ) -> str:
        return FAKE_DIFF

    async def get_pr_details(
        self, owner: str, repo: str, pr_number: int
    ) -> Dict[str, Any]:
        return FAKE_PR_DETAILS

    async def get_pr_head_commit(
        self, owner: str, repo: str, pr_number: int
    ) -> str:
        return FAKE_PR_DETAILS["head_sha"]

    async def get_file_content(
        self, owner: str, repo: str, path: str, ref: str
    ) -> Optional[str]:
        return "x = 1\ny = None\nprint(y.upper())\n"

    async def get_authenticated_user_login(self) -> str:
        return "dummy-bot"

    async def create_review_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_id: str,
        path: str,
        line: int,
        body: str,
        side: str = "RIGHT",
    ) -> None:
        pass

    async def create_comment(
        self, owner: str, repo: str, pr_number: int, body: str
    ) -> int:
        self.posted_comments.append(body)
        return 42

    async def create_pull_request_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_id: str,
        body: str,
        comments: List[Dict[str, Any]],
    ) -> int:
        self.posted_reviews.append({"body": body, "comments": comments})
        return 99

    async def set_commit_status(
        self,
        owner: str,
        repo: str,
        sha: str,
        state: str,
        description: str,
        context: str = "sift/review",
    ) -> None:
        self.set_statuses.append((state, description, context))

    async def get_comment_reactions(
        self, owner: str, repo: str, comment_id: int
    ) -> list:
        return []

    async def get_pull_request_review_reactions(
        self, owner: str, repo: str, pr_number: int, review_id: int
    ) -> list:
        return []

    async def list_pull_request_review_comments(
        self, owner: str, repo: str, pr_number: int
    ) -> list:
        return []

    async def get_review_comment_reactions(
        self, owner: str, repo: str, comment_id: int
    ) -> list:
        return []


def test_register_and_get_forge() -> None:
    """Registry stores and retrieves ForgeProvider classes by key."""
    register_forge("dummy", DummyForge)
    assert get_forge("dummy") is DummyForge


def test_get_forge_unknown_key_raises() -> None:
    from sift.integrations.registry import get_forge
    with pytest.raises(KeyError, match="no_such_forge"):
        get_forge("no_such_forge")


@pytest.mark.asyncio
async def test_dummy_forge_drives_run_review() -> None:
    """DummyForge drives run_review end-to-end without touching any GitHub code."""
    forge = DummyForge()

    fake_finding = {
        "line": 3,
        "body": FAKE_COMMENT_BODY,
        "post_inline": True,
        "severity": "bug",
        "title": "Null dereference",
        "confidence": 9,
        "fix": None,
    }

    with (
        patch("sift.intelligence.passes.candidates.llm_client.review_file", new=AsyncMock(return_value=[fake_finding])),
        patch("sift.intelligence.passes.critic.critique", new=_critic_passthrough),
        patch("sift.core.review_engine.summarize_review", new=AsyncMock(return_value="Summary of review.")),
        patch("sift.core.review_engine.run_linters", return_value={}),
        patch("sift.core.review_engine.run_semgrep", return_value={}),
        patch("sift.core.review_engine.scan_diff_for_secrets", return_value=[]),
        patch("sift.core.review_engine.store_review", return_value=1),
        patch("sift.core.review_engine.get_repo_feedback_comment_examples", return_value=[]),
        patch("sift.core.review_engine.get_avg_quality_score_for_path_pattern", return_value=None),
        patch("sift.core.review_engine.get_tool_cache_hits", return_value={}),
        patch("sift.core.review_engine.store_tool_cache"),
        patch("sift.core.review_engine.resolve_pr_import_graph", return_value={}),
    ):
        from sift.core.review_engine import run_review

        await run_review(lambda: forge, "owner", "repo", 1)

    # A summary comment was posted via the forge
    assert len(forge.posted_comments) == 1
    assert forge.posted_comments[0]  # non-empty

    # Inline review was posted (findings were present)
    assert len(forge.posted_reviews) == 1
    review_comments = forge.posted_reviews[0]["comments"]
    assert len(review_comments) >= 1
    assert review_comments[0]["path"] == "app/calc.py"


MULTI_FILE_DIFF = """\
diff --git a/app/calc.py b/app/calc.py
--- a/app/calc.py
+++ b/app/calc.py
@@ -1,2 +1,3 @@
 x = 1
+y = None
+print(y.upper())
diff --git a/app/flaky.py b/app/flaky.py
--- a/app/flaky.py
+++ b/app/flaky.py
@@ -1,2 +1,3 @@
 x = 1
+z = 2
+print(z)
diff --git a/app/broken.py b/app/broken.py
--- a/app/broken.py
+++ b/app/broken.py
@@ -1,2 +1,3 @@
 x = 1
+w = 3
+print(w)
diff --git a/yarn.lock b/yarn.lock
--- a/yarn.lock
+++ b/yarn.lock
@@ -1,2 +1,3 @@
 # yarn lockfile v1
+left-pad@1.3.0:
+  version "1.3.0"
"""


class MultiFileForge(DummyForge):
    async def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        return MULTI_FILE_DIFF


@pytest.mark.asyncio
async def test_failed_files_are_retried_then_reported_unreviewed() -> None:
    """A file that fails once is recovered by the sequential retry pass; one that keeps
    failing is reported as unreviewed (never as "no issues"); lockfiles are skipped."""
    forge = MultiFileForge()
    calls: Dict[str, int] = {}

    async def _pipeline(file_input, *args, **kwargs):
        calls[file_input.path] = calls.get(file_input.path, 0) + 1
        if file_input.path == "app/broken.py":
            raise RuntimeError("429 too many concurrent requests")
        if file_input.path == "app/flaky.py" and calls[file_input.path] == 1:
            raise RuntimeError("429 too many concurrent requests")
        return []

    summarize = AsyncMock(return_value="Summary of review.")
    store = MagicMock(return_value=1)
    with (
        patch("sift.core.review_engine.run_pipeline_per_file", new=_pipeline),
        patch("sift.core.review_engine.summarize_review", new=summarize),
        patch("sift.core.review_engine.run_linters", return_value={}),
        patch("sift.core.review_engine.run_semgrep", return_value={}),
        patch("sift.core.review_engine.scan_diff_for_secrets", return_value=[]),
        patch("sift.core.review_engine.store_review", new=store),
        patch("sift.core.review_engine.get_repo_feedback_comment_examples", return_value=[]),
        patch("sift.core.review_engine.get_avg_quality_score_for_path_pattern", return_value=None),
        patch("sift.core.review_engine.get_tool_cache_hits", return_value={}),
        patch("sift.core.review_engine.store_tool_cache"),
        patch("sift.core.review_engine.resolve_pr_import_graph", return_value={}),
    ):
        from sift.core.review_engine import run_review

        await run_review(lambda: forge, "owner", "repo", 1)

    assert calls == {"app/calc.py": 1, "app/flaky.py": 2, "app/broken.py": 2}
    assert "yarn.lock" not in calls
    kwargs = summarize.await_args.kwargs
    assert kwargs["unreviewed"] == ["app/broken.py"]
    assert [p for p, _ in kwargs["skipped"]] == ["yarn.lock"]
    stored_paths = store.call_args.kwargs["paths"]
    assert sorted(stored_paths) == ["app/calc.py", "app/flaky.py"]


@pytest.mark.asyncio
async def test_permanent_llm_errors_are_not_retried_and_report_a_reason() -> None:
    """A deterministic 4xx (e.g. context length) skips the sequential retry pass and is
    reported with its reason; a 429 is still retried."""
    import litellm

    forge = MultiFileForge()
    calls: Dict[str, int] = {}

    async def _pipeline(file_input, *args, **kwargs):
        calls[file_input.path] = calls.get(file_input.path, 0) + 1
        if file_input.path == "app/broken.py":
            raise litellm.BadRequestError(
                message="OpenAIException - The prompt is too long: 147021, model maximum context length: 131072",
                model="openai/x",
                llm_provider="openai",
            )
        if file_input.path == "app/flaky.py" and calls[file_input.path] == 1:
            raise RuntimeError("429 too many concurrent requests")
        return []

    summarize = AsyncMock(return_value="Summary of review.")
    with (
        patch("sift.core.review_engine.run_pipeline_per_file", new=_pipeline),
        patch("sift.core.review_engine.summarize_review", new=summarize),
        patch("sift.core.review_engine.run_linters", return_value={}),
        patch("sift.core.review_engine.run_semgrep", return_value={}),
        patch("sift.core.review_engine.scan_diff_for_secrets", return_value=[]),
        patch("sift.core.review_engine.store_review", new=MagicMock(return_value=1)),
        patch("sift.core.review_engine.get_repo_feedback_comment_examples", return_value=[]),
        patch("sift.core.review_engine.get_avg_quality_score_for_path_pattern", return_value=None),
        patch("sift.core.review_engine.get_tool_cache_hits", return_value={}),
        patch("sift.core.review_engine.store_tool_cache"),
        patch("sift.core.review_engine.resolve_pr_import_graph", return_value={}),
    ):
        from sift.core.review_engine import run_review

        await run_review(lambda: forge, "owner", "repo", 1)

    assert calls == {"app/calc.py": 1, "app/flaky.py": 2, "app/broken.py": 1}
    kwargs = summarize.await_args.kwargs
    assert kwargs["unreviewed"] == ["app/broken.py"]
    assert kwargs["unreviewed_reasons"] == {"app/broken.py": "prompt exceeds the model's context window"}
