"""GitHub API client: App JWT, installation token, PR diff, post comment."""
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
        """Fetch PR metadata (title, body) for context."""
        if not self._client:
            raise RuntimeError("GitHubClient must be used as async context manager")
        r = await self._client.get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
        r.raise_for_status()
        data = r.json()
        return {"title": data.get("title") or "", "body": data.get("body") or ""}

    async def create_comment(self, owner: str, repo: str, pr_number: int, body: str) -> None:
        """Post a comment on a PR (issue comment)."""
        if not self._client:
            raise RuntimeError("GitHubClient must be used as async context manager")
        r = await self._client.post(
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            json={"body": body},
        )
        r.raise_for_status()
        logger.info("Posted comment on %s/%s PR #%s", owner, repo, pr_number)


async def get_installation_token(installation_id: int) -> str:
    """Public helper to get token for an installation (e.g. for review engine)."""
    return await _get_installation_token(installation_id)
