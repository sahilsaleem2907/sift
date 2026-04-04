"""GitHub API client: installation token, PR diff, post comment."""
import base64
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from src import config

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"


async def _get_installation_token(installation_id: int) -> str:
    """Resolve a GitHub token for an installation via external token service or static fallback."""
    if config.SWIFT_API_BACKEND_BASE_URL:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if config.SIFT_API_KEY:
            headers["Authorization"] = f"Bearer {config.SIFT_API_KEY}"
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{config.SWIFT_API_BACKEND_BASE_URL}/api/github/installation-token",
                json={"installation_id": installation_id},
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
        token = data.get("token")
        if not isinstance(token, str) or not token:
            raise RuntimeError("Token service did not return a valid 'token' field")
        return token
    if config.SIFT_GITHUB_TOKEN:
        return config.SIFT_GITHUB_TOKEN
    raise RuntimeError(
        "Cannot resolve GitHub token: set SWIFT_API_BACKEND_BASE_URL or SIFT_GITHUB_TOKEN"
    )


class GitHubClient:
    """Async client for GitHub API using an installation access token."""

    def __init__(self, installation_id: int, token: Optional[str] = None):
        self._installation_id = installation_id
        self._token = token
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "GitHubClient":
        if self._token is None:
            self._token = await _get_installation_token(self._installation_id)
        self._client = httpx.AsyncClient(
            base_url=GITHUB_API_BASE,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
            },
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _paginate_get(
        self, url: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None
    ) -> List[Any]:
        """GET a paginated list endpoint, following Link rel=next until exhausted."""
        if not self._client:
            raise RuntimeError("GitHubClient must be used as async context manager")
        merged_params: Dict[str, Any] = {"per_page": 100, **(params or {})}
        results: List[Any] = []
        next_url: Optional[str] = url
        while next_url:
            r = await self._client.get(next_url, params=merged_params, headers=headers)
            r.raise_for_status()
            results.extend(r.json())
            link_header = r.headers.get("link", "")
            match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
            next_url = match.group(1) if match else None
            merged_params = {}
        return results

    async def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Fetch PR diff as raw text."""
        if not self._client:
            raise RuntimeError("GitHubClient must be used as async context manager")
        r = await self._client.get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}",
            headers={"Accept": "application/vnd.github.v3.diff"},
        )
        r.raise_for_status()
        return r.text

    async def get_compare_diff(
        self, owner: str, repo: str, base_sha: str, head_sha: str
    ) -> str:
        """Fetch diff between two commits (e.g. previous head and current head for incremental review)."""
        if not self._client:
            raise RuntimeError("GitHubClient must be used as async context manager")
        r = await self._client.get(
            f"/repos/{owner}/{repo}/compare/{base_sha}...{head_sha}",
            headers={"Accept": "application/vnd.github.v3.diff"},
        )
        r.raise_for_status()
        return r.text

    async def get_pr_details(self, owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
        """Fetch PR metadata (title, body, head_sha) for context."""
        if not self._client:
            raise RuntimeError("GitHubClient must be used as async context manager")
        r = await self._client.get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
        r.raise_for_status()
        data = r.json()
        head = data.get("head") or {}
        head_sha = (head.get("sha") or "") if isinstance(head, dict) else ""
        return {
            "title": data.get("title") or "",
            "body": data.get("body") or "",
            "head_sha": head_sha,
        }

    async def get_pr_head_commit(self, owner: str, repo: str, pr_number: int) -> str:
        """Return the PR head commit SHA (for posting review comments)."""
        details = await self.get_pr_details(owner, repo, pr_number)
        sha = details.get("head_sha") or ""
        if not sha:
            raise ValueError("PR head SHA not found")
        return sha

    async def get_file_content(
        self, owner: str, repo: str, path: str, ref: str
    ) -> Optional[str]:
        """Fetch file content at ref (e.g. commit SHA). Returns None for 404, binary, or decode errors."""
        if not self._client:
            raise RuntimeError("GitHubClient must be used as async context manager")
        try:
            r = await self._client.get(
                f"/repos/{owner}/{repo}/contents/{path}",
                params={"ref": ref},
            )
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("type") == "file" and "content" in data:
                content_b64 = (data["content"] or "").replace("\n", "").strip()
                try:
                    return base64.b64decode(content_b64).decode("utf-8")
                except (ValueError, UnicodeDecodeError):
                    return None
            return None
        except httpx.HTTPError:
            return None

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
        """Post an inline review comment on the PR diff (Files changed tab).

        commit_id: PR head commit SHA. path: file path. line: line in new file. side: RIGHT for new file.
        """
        if not self._client:
            raise RuntimeError("GitHubClient must be used as async context manager")
        payload = {
            "commit_id": commit_id,
            "path": path,
            "line": line,
            "side": side,
            "body": body,
        }
        r = await self._client.post(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/comments",
            json=payload,
        )
        if r.status_code == 422:
            logger.warning(
                "Review comment rejected (422) for %s/%s PR #%s path=%s line=%s: %s",
                owner,
                repo,
                pr_number,
                path,
                line,
                r.text,
            )
            return
        r.raise_for_status()
        logger.info(
            "Posted review comment on %s/%s PR #%s path=%s line=%s",
            owner,
            repo,
            pr_number,
            path,
            line,
        )

    async def create_comment(self, owner: str, repo: str, pr_number: int, body: str) -> int:
        """Post a comment on a PR (issue comment). Returns the created comment id."""
        if not self._client:
            raise RuntimeError("GitHubClient must be used as async context manager")
        r = await self._client.post(
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            json={"body": body},
        )
        r.raise_for_status()
        comment_id = r.json()["id"]
        logger.info("Posted comment on %s/%s PR #%s (id=%s)", owner, repo, pr_number, comment_id)
        return comment_id

    async def create_pull_request_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_id: str,
        body: str,
        comments: List[Dict[str, Any]],
    ) -> int:
        """Post all inline comments + summary in a single Reviews API call. Returns the review id."""
        if not self._client:
            raise RuntimeError("GitHubClient must be used as async context manager")
        payload = {
            "commit_id": commit_id,
            "body": body,
            "event": "COMMENT",
            "comments": [
                {"path": c["path"], "line": c["line"], "side": "RIGHT", "body": c["body"]}
                for c in comments
            ],
        }
        r = await self._client.post(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
            json=payload,
        )
        r.raise_for_status()
        review_id = r.json()["id"]
        logger.info(
            "Posted pull request review on %s/%s PR #%s (id=%s, %d comment(s))",
            owner, repo, pr_number, review_id, len(comments),
        )
        return review_id

    async def get_authenticated_user_login(self) -> str:
        """Return the login for the authenticated user/bot."""
        if not self._client:
            raise RuntimeError("GitHubClient must be used as async context manager")
        r = await self._client.get("/user")
        r.raise_for_status()
        data = r.json()
        login = (data.get("login") or "").strip()
        if not login:
            raise ValueError("GET /user did not return login")
        return login

    async def get_comment_reactions(
        self, owner: str, repo: str, comment_id: int
    ) -> list:
        """List reactions on an issue comment. Returns list of dicts with user.login and content."""
        return await self._paginate_get(
            f"/repos/{owner}/{repo}/issues/comments/{comment_id}/reactions",
            headers={"Accept": "application/vnd.github+json"},
        )

    async def get_pull_request_review_reactions(
        self, owner: str, repo: str, pr_number: int, review_id: int
    ) -> list:
        """List reactions on a pull request review summary (from POST .../pulls/{n}/reviews)."""
        return await self._paginate_get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews/{review_id}/reactions",
            headers={"Accept": "application/vnd.github+json"},
        )

    async def list_pull_request_review_comments(
        self, owner: str, repo: str, pr_number: int
    ) -> list:
        """List all pull request review comments (inline) on the PR. Returns list of dicts with id, user.login, etc."""
        return await self._paginate_get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/comments"
        )

    async def get_review_comment_reactions(
        self, owner: str, repo: str, comment_id: int
    ) -> list:
        """List reactions on a pull request review comment (inline). Returns list of dicts with user.login and content."""
        return await self._paginate_get(
            f"/repos/{owner}/{repo}/pulls/comments/{comment_id}/reactions",
            headers={"Accept": "application/vnd.github+json"},
        )


async def get_installation_token(installation_id: int) -> str:
    """Public helper to get token for an installation (e.g. for review engine)."""
    return await _get_installation_token(installation_id)
