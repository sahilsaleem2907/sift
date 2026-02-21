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
