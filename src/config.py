"""Centralized configuration from environment."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Required
DATABASE_URL = os.environ.get("DATABASE_URL")

# Optional: bearer token to protect POST /review (GitHub Actions flow) and authenticate to token service; skip auth if unset
SIFT_API_KEY = os.environ.get("SIFT_API_KEY") or None
SWIFT_API_BACKEND_BASE_URL = os.environ.get("SWIFT_API_BACKEND_BASE_URL")
SIFT_GITHUB_TOKEN = os.environ.get("SIFT_GITHUB_TOKEN") or None

# LLM provider (LiteLLM): model string and optional api_base
LLM_MODEL = os.environ.get("LLM_MODEL", "ollama/llama3.2")
# Only set when explicitly provided (Ollama/Azure); leave None for OpenAI, Anthropic, Gemini, etc.
_llm_base = os.environ.get("LLM_API_BASE")
LLM_API_BASE = _llm_base.rstrip("/") if _llm_base else None

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
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "ollama/nomic-embed-text")
# Only set when explicitly provided (Ollama/custom); leave None for OpenAI, Gemini, etc.
_embed_base = os.environ.get("EMBEDDING_API_BASE")
EMBEDDING_API_BASE = _embed_base.rstrip("/") if _embed_base else None
# Dimension for embedding vectors (nomic-embed-text = 768; required for HNSW index)
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION") or "768")
VECTOR_SIMILARITY_TOP_K = int(os.environ.get("VECTOR_SIMILARITY_TOP_K") or "5")
_VECTOR_EXCLUDE_SAME_FILE_RAW = (os.environ.get("VECTOR_EXCLUDE_SAME_FILE") or "1").strip().lower()
VECTOR_EXCLUDE_SAME_FILE = _VECTOR_EXCLUDE_SAME_FILE_RAW in ("1", "true", "yes")

# Concurrency / rate limiting
SIFT_MAX_CONCURRENT_REVIEWS = int(os.environ.get("SIFT_MAX_CONCURRENT_REVIEWS") or "10")
SIFT_LLM_REQUEST_DELAY = float(os.environ.get("SIFT_LLM_REQUEST_DELAY") or "0.5")
SIFT_GITHUB_COMMENT_DELAY = float(os.environ.get("SIFT_GITHUB_COMMENT_DELAY") or "0.2")


def validate_required() -> None:
    """Fail fast if required env vars are missing."""
    _log = logging.getLogger("src.config")
    if not DATABASE_URL:
        raise RuntimeError("Missing required environment variable: DATABASE_URL")
    if not SWIFT_API_BACKEND_BASE_URL and not SIFT_GITHUB_TOKEN:
        _log.warning(
            "Neither SWIFT_API_BACKEND_BASE_URL nor SIFT_GITHUB_TOKEN is set; "
            "GitHub integration will not work for installation_id auth mode."
        )


def setup_logging() -> None:
    """Configure global logging once at app startup."""
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


