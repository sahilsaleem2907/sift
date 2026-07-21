"""Abstract base class for forge (SCM) providers."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ForgeProvider(ABC):
    """Interface every forge integration must implement.

    Concrete providers (GitHub, Bitbucket, …) subclass this and implement the
    methods below. Return types are kept as raw dicts/primitives for now;
    normalized models will be added when a second provider validates the shape.

    All providers must support the async context manager protocol so the
    underlying HTTP client lifecycle is managed correctly.
    """

    # -- async context manager --

    @abstractmethod
    async def __aenter__(self) -> "ForgeProvider":
        ...

    @abstractmethod
    async def __aexit__(self, *args: Any) -> None:
        ...

    # -- read operations --

    @abstractmethod
    async def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Fetch the PR diff as raw unified-diff text."""
        ...

    @abstractmethod
    async def get_compare_diff(
        self, owner: str, repo: str, base_sha: str, head_sha: str
    ) -> str:
        """Fetch diff between two commits."""
        ...

    @abstractmethod
    async def get_pr_details(
        self, owner: str, repo: str, pr_number: int
    ) -> Dict[str, Any]:
        """Return PR metadata dict with at least: title, body, head_sha."""
        ...

    @abstractmethod
    async def get_pr_head_commit(
        self, owner: str, repo: str, pr_number: int
    ) -> str:
        """Return the PR head commit SHA."""
        ...

    @abstractmethod
    async def get_file_content(
        self, owner: str, repo: str, path: str, ref: str
    ) -> Optional[str]:
        """Fetch file content at ref. Returns None for 404 / binary / decode errors."""
        ...

    @abstractmethod
    async def get_authenticated_user_login(self) -> str:
        """Return the login name of the authenticated bot/user."""
        ...

    # -- write / post operations --

    @abstractmethod
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
        """Post an inline review comment on the PR diff."""
        ...

    @abstractmethod
    async def create_comment(
        self, owner: str, repo: str, pr_number: int, body: str
    ) -> int:
        """Post a PR-level (issue) comment. Returns the created comment id."""
        ...

    @abstractmethod
    async def create_pull_request_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_id: str,
        body: str,
        comments: List[Dict[str, Any]],
    ) -> int:
        """Post inline comments + summary in a single batched review. Returns review id."""
        ...

    @abstractmethod
    async def set_commit_status(
        self,
        owner: str,
        repo: str,
        sha: str,
        state: str,
        description: str,
        context: str = "sift/review",
    ) -> None:
        """Post a build/commit status (pending | success | failure | error)."""
        ...

    # -- optional helpers --

    def get_clone_token(self) -> Optional[str]:
        """Return an auth token suitable for authenticated git clones, or None.

        Providers that support clone authentication should override this.
        The review engine uses this to pass credentials to get_repo_at_commit.
        """
        return None

    # -- reaction / feedback read operations --

    @abstractmethod
    async def get_comment_reactions(
        self, owner: str, repo: str, comment_id: int
    ) -> list:
        """List reactions on an issue comment."""
        ...

    @abstractmethod
    async def get_pull_request_review_reactions(
        self, owner: str, repo: str, pr_number: int, review_id: int
    ) -> list:
        """List reactions on a pull request review summary."""
        ...

    @abstractmethod
    async def list_pull_request_review_comments(
        self, owner: str, repo: str, pr_number: int
    ) -> list:
        """List all inline review comments on a PR."""
        ...

    @abstractmethod
    async def get_review_comment_reactions(
        self, owner: str, repo: str, comment_id: int
    ) -> list:
        """List reactions on a single inline review comment."""
        ...
