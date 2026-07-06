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
SIFT_API_BACKEND_BASE_URL = os.environ.get("SIFT_API_BACKEND_BASE_URL")
SIFT_GITHUB_TOKEN = os.environ.get("SIFT_GITHUB_TOKEN") or None
GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET")

# LLM provider (LiteLLM): model string and optional api_base / api_key
LLM_MODEL = os.environ.get("LLM_MODEL", "ollama/llama3.2")
# Custom base URL for any LiteLLM provider that needs it (e.g. Ollama, Azure OpenAI, self-hosted
# gateways). Prefer LLM_API_BASE; else SIFT_LLM_API_BASE (GitHub Actions secret passthrough).
_llm_base = (os.environ.get("LLM_API_BASE") or "").strip() or (os.environ.get("SIFT_LLM_API_BASE") or "").strip()
LLM_API_BASE = _llm_base.rstrip("/") if _llm_base else None
# Explicit API key for the primary model. When set, passed directly to LiteLLM so it takes
# precedence over provider-specific env vars (needed when using a custom api_base like OpenRouter).
LLM_API_KEY = (os.environ.get("LLM_API_KEY") or "").strip() or None

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Semgrep (optional; default enabled)
_SEMGREP_ENABLED_RAW = (os.environ.get("SEMGREP_ENABLED") or "1").strip().lower()
SEMGREP_ENABLED = _SEMGREP_ENABLED_RAW in ("1", "true", "yes")

# CodeQL (optional; default disabled)
_CODEQL_ENABLED_RAW = (os.environ.get("CODEQL_ENABLED") or "0").strip().lower()
CODEQL_ENABLED = _CODEQL_ENABLED_RAW in ("1", "true", "yes")
CODEQL_SUITE_RAW = (os.environ.get("CODEQL_SUITE") or "default").strip().lower()
_VALID_SUITES = ("default", "security-extended", "security-and-quality")
CODEQL_SUITE = CODEQL_SUITE_RAW if CODEQL_SUITE_RAW in _VALID_SUITES else "default"
CODEQL_TIMEOUT = int(os.environ.get("CODEQL_TIMEOUT") or "600")

# Pyright type-checker (optional; default disabled). Deterministic API/version-existence
# findings, promoted critic-exempt. Runs against the full clone (like CodeQL).
_PYRIGHT_ENABLED_RAW = (os.environ.get("PYRIGHT_ENABLED") or "0").strip().lower()
PYRIGHT_ENABLED = _PYRIGHT_ENABLED_RAW in ("1", "true", "yes")
PYRIGHT_TIMEOUT = int(os.environ.get("PYRIGHT_TIMEOUT") or "600")

# Deterministic AST verdict analyzers (language seam). High-precision, promoted
# as critic-exempt findings. On by default; set ANALYZERS_ENABLED=0 to disable.
_ANALYZERS_ENABLED_RAW = (os.environ.get("ANALYZERS_ENABLED") or "1").strip().lower()
ANALYZERS_ENABLED = _ANALYZERS_ENABLED_RAW in ("1", "true", "yes")
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

# JS/TS ecosystem tools (default: all enabled)
_OXLINT_RAW = (os.environ.get("OXLINT_ENABLED") or "1").strip().lower()
OXLINT_ENABLED = _OXLINT_RAW in ("1", "true", "yes")
_NPM_AUDIT_RAW = (os.environ.get("NPM_AUDIT_ENABLED") or "1").strip().lower()
NPM_AUDIT_ENABLED = _NPM_AUDIT_RAW in ("1", "true", "yes")
_YARN_AUDIT_RAW = (os.environ.get("YARN_AUDIT_ENABLED") or "1").strip().lower()
YARN_AUDIT_ENABLED = _YARN_AUDIT_RAW in ("1", "true", "yes")
_SEMGREP_FRAMEWORK_RAW = (os.environ.get("SEMGREP_FRAMEWORK_RULES_ENABLED") or "1").strip().lower()
SEMGREP_FRAMEWORK_RULES_ENABLED = _SEMGREP_FRAMEWORK_RAW in ("1", "true", "yes")

# Smart analysis routing (optional; default enabled)
_SMART_ROUTING_RAW = (os.environ.get("SIFT_SMART_ROUTING_ENABLED") or "1").strip().lower()
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

# PR blocking via commit status (default: disabled)
_BLOCK_PRS_RAW = (os.environ.get("SIFT_BLOCK_PRS_ENABLED") or "0").strip().lower()
SIFT_BLOCK_PRS_ENABLED = _BLOCK_PRS_RAW in ("1", "true", "yes")
SIFT_BLOCK_ON_SEVERITIES = [
    s.strip().lower()
    for s in (os.environ.get("SIFT_BLOCK_ON_SEVERITIES") or "bug,security").split(",")
    if s.strip()
]
SIFT_BLOCK_MIN_FINDINGS = int(os.environ.get("SIFT_BLOCK_MIN_FINDINGS") or "1")
SIFT_STATUS_CONTEXT = os.environ.get("SIFT_STATUS_CONTEXT") or "sift/review"

# Review engine effort: low | balanced | high  (default: balanced)
SIFT_REVIEW_EFFORT = (os.environ.get("SIFT_REVIEW_EFFORT") or "balanced").strip().lower()

# Optional separate model for critic / holistic passes (defaults to LLM_MODEL when unset).
SIFT_REVIEW_MODEL = os.environ.get("SIFT_REVIEW_MODEL") or None
SIFT_REVIEW_MODEL_KEY = os.environ.get("SIFT_REVIEW_MODEL_KEY") or None
_review_base = (os.environ.get("SIFT_REVIEW_MODEL_BASE_URL") or "").strip()
SIFT_REVIEW_MODEL_BASE_URL = _review_base.rstrip("/") if _review_base else None

# JSON object to hard-override capability detection for unknown / self-hosted models.
# SIFT_CAPABILITY_OVERRIDE applies to the primary model; SIFT_REVIEW_CAPABILITY_OVERRIDE to
# the critic/holistic model. Both unset by default so detection runs (per-role, not global).
SIFT_CAPABILITY_OVERRIDE = (os.environ.get("SIFT_CAPABILITY_OVERRIDE") or "").strip() or None
SIFT_REVIEW_CAPABILITY_OVERRIDE = (os.environ.get("SIFT_REVIEW_CAPABILITY_OVERRIDE") or "").strip() or None

# Max tool-call steps in the high-effort agentic retrieval loop (Phase 4).
SIFT_AGENTIC_MAX_STEPS = int(os.environ.get("SIFT_AGENTIC_MAX_STEPS") or "4")

# Per-file reviewer: render the whole file as context when it is at or under this
# many lines; above it, fall back to rendering only the changed line ranges.
# Whole-file context lets the model verify cross-references (e.g. that an import is
# used elsewhere) instead of guessing from excerpts.
SIFT_FULL_FILE_RENDER_MAX_LINES = int(os.environ.get("SIFT_FULL_FILE_RENDER_MAX_LINES") or "800")


def validate_required() -> None:
    """Fail fast if required env vars are missing."""
    _log = logging.getLogger("src.config")
    if not DATABASE_URL:
        raise RuntimeError("Missing required environment variable: DATABASE_URL")
    if not SIFT_API_BACKEND_BASE_URL and not SIFT_GITHUB_TOKEN:
        _log.warning(
            "Neither SIFT_API_BACKEND_BASE_URL nor SIFT_GITHUB_TOKEN is set; "
            "GitHub integration will not work for installation_id auth mode."
        )
    _valid_efforts = ("low", "balanced", "high")
    if SIFT_REVIEW_EFFORT not in _valid_efforts:
        _log.warning(
            "SIFT_REVIEW_EFFORT=%r is not one of %s; falling back to 'balanced'.",
            SIFT_REVIEW_EFFORT,
            _valid_efforts,
        )


def setup_logging() -> None:
    """Configure global logging once at app startup."""
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


