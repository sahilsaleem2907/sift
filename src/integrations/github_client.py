"""GitHub API client: App JWT, installation token, PR diff, post comment."""
import base64
import logging
import time
from typing import Any, Dict, Optional

import httpx
import jwt

from src import config

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"


def _make_jwt() -> str:
    """Generate a JWT for the GitHub App (RS256, iat/exp per GitHub docs)."""
    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + 600,
        "iss": config.GITHUB_APP_ID,
    }
    key = config.get_github_private_key_bytes()
    return jwt.encode(payload, key, algorithm="RS256")  # type: ignore[return-value]


async def _get_installation_token(installation_id: int) -> str:
    """Exchange App JWT for an installation access token; cache not implemented (simple)."""
    token = _make_jwt()
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{GITHUB_API_BASE}/app/installations/{installation_id}/access_tokens",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
            },
        )
        r.raise_for_status()
        data = r.json()
    return data["token"]


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

    async def get_comment_reactions(
        self, owner: str, repo: str, comment_id: int
    ) -> list:
        """List reactions on an issue comment. Returns list of dicts with user.login and content."""
        if not self._client:
            raise RuntimeError("GitHubClient must be used as async context manager")
        r = await self._client.get(
            f"/repos/{owner}/{repo}/issues/comments/{comment_id}/reactions",
            headers={"Accept": "application/vnd.github+json"},
        )
        r.raise_for_status()
        return r.json()


async def get_installation_token(installation_id: int) -> str:
    """Public helper to get token for an installation (e.g. for review engine)."""
    return await _get_installation_token(installation_id)
