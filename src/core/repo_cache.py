"""Git clone cache: first PR clones, subsequent PRs fetch + checkout. Used for CodeQL."""
import logging
import subprocess
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

CLONE_FETCH_TIMEOUT = 300


def get_repo_at_commit(owner: str, repo: str, commit_sha: str, token: str) -> Path:
    """Return path to repo checkout at the given commit.

    Uses cache at config.SIFT_CLONE_CACHE_DIR / owner / repo.
    If cache miss: clone then checkout. If cache hit: fetch origin then checkout.
    Raises on git failure (caller should catch and skip CodeQL).
    """
    cache_dir = config.SIFT_CLONE_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_path = cache_dir / owner / repo

    url = f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"

    if not repo_path.exists() or not (repo_path / ".git").exists():
        logger.info("Cloning %s/%s for CodeQL cache", owner, repo)
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", url, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=CLONE_FETCH_TIMEOUT,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", commit_sha],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(repo_path),
            check=True,
        )
        logger.debug("Clone and checkout %s at %s", repo_path, commit_sha)
        return repo_path

    logger.debug("Fetching and checking out %s at %s", repo_path, commit_sha)
    subprocess.run(
        ["git", "fetch", "origin"],
        capture_output=True,
        text=True,
        timeout=CLONE_FETCH_TIMEOUT,
        cwd=str(repo_path),
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--force", commit_sha],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(repo_path),
        check=True,
    )
    return repo_path
