"""Centralized configuration from environment."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Required
GITHUB_APP_ID = os.environ.get("GITHUB_APP_ID")
GITHUB_APP_PRIVATE_KEY = os.environ.get("GITHUB_APP_PRIVATE_KEY")
GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET")
DATABASE_URL = os.environ.get("DATABASE_URL")

# Ollama (optional, have defaults)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# CodeQL (optional; default disabled)
_CODEQL_ENABLED_RAW = (os.environ.get("CODEQL_ENABLED") or "0").strip().lower()
CODEQL_ENABLED = _CODEQL_ENABLED_RAW in ("1", "true", "yes")
CODEQL_SUITE_RAW = (os.environ.get("CODEQL_SUITE") or "default").strip().lower()
_VALID_SUITES = ("default", "security-extended", "security-and-quality")
CODEQL_SUITE = CODEQL_SUITE_RAW if CODEQL_SUITE_RAW in _VALID_SUITES else "default"
CODEQL_TIMEOUT = int(os.environ.get("CODEQL_TIMEOUT") or "600")
# Base directory for cached git clones (default: ~/.sift/clones or temp)
_sift_clones = os.environ.get("SIFT_CLONE_CACHE_DIR")
if _sift_clones:
    SIFT_CLONE_CACHE_DIR = Path(_sift_clones).expanduser().resolve()
else:
    SIFT_CLONE_CACHE_DIR = Path.home() / ".sift" / "clones"

# Tool result cache (Semgrep, linter, CodeQL); default enabled, 24h TTL
_TOOL_CACHE_RAW = (os.environ.get("TOOL_CACHE_ENABLED") or "1").strip().lower()
TOOL_CACHE_ENABLED = _TOOL_CACHE_RAW in ("1", "true", "yes")
TOOL_CACHE_TTL_HOURS = int(os.environ.get("TOOL_CACHE_TTL_HOURS") or "24")

# Smart analysis routing (optional; default disabled)
_SMART_ROUTING_RAW = (os.environ.get("SIFT_SMART_ROUTING_ENABLED") or "0").strip().lower()
SIFT_SMART_ROUTING_ENABLED = _SMART_ROUTING_RAW in ("1", "true", "yes")

# Vector DB / code similarity (optional; default disabled)
_VECTOR_DB_ENABLED_RAW = (os.environ.get("VECTOR_DB_ENABLED") or "0").strip().lower()
VECTOR_DB_ENABLED = _VECTOR_DB_ENABLED_RAW in ("1", "true", "yes")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
# Dimension for embedding vectors (nomic-embed-text = 768; required for HNSW index)
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION") or "768")
VECTOR_SIMILARITY_TOP_K = int(os.environ.get("VECTOR_SIMILARITY_TOP_K") or "5")
_VECTOR_EXCLUDE_SAME_FILE_RAW = (os.environ.get("VECTOR_EXCLUDE_SAME_FILE") or "1").strip().lower()
VECTOR_EXCLUDE_SAME_FILE = _VECTOR_EXCLUDE_SAME_FILE_RAW in ("1", "true", "yes")


def validate_required() -> None:
    """Fail fast if required env vars are missing."""
    missing = []
    if not GITHUB_APP_ID:
        missing.append("GITHUB_APP_ID")
    if not GITHUB_APP_PRIVATE_KEY:
        missing.append("GITHUB_APP_PRIVATE_KEY")
    if not GITHUB_WEBHOOK_SECRET:
        missing.append("GITHUB_WEBHOOK_SECRET")
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def setup_logging() -> None:
    """Configure global logging once at app startup."""
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def get_github_private_key_bytes() -> bytes:
    """Return GitHub App private key as bytes (from env or file path)."""
    raw = GITHUB_APP_PRIVATE_KEY or ""
    raw = raw.strip()
    if raw.startswith("-----BEGIN"):
        return raw.encode("utf-8")
    path = Path(raw)
    if path.exists():
        return path.read_bytes()
    raise FileNotFoundError(f"GITHUB_APP_PRIVATE_KEY is not PEM and path does not exist: {raw}")
