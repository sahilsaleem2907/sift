#!/usr/bin/env bash
# Benchmark "peak" config overlay. Source AFTER .env:
#   set -a; source .env; source scripts/bench_env.sh; set +a
# Overrides only the LLM models + asserts peak toggles; everything else (DB, secrets,
# semgrep/linter flags) is inherited from .env. config.py's load_dotenv(override=False)
# means these shell exports win over any .env value.

# DeepSeek V4 via OpenRouter — Flash generates per-file, Pro critiques/holistic.
export LLM_API_BASE="https://openrouter.ai/api/v1"
export LLM_API_KEY="${OPENROUTER_API_KEY:?source .env first so OPENROUTER_API_KEY is set}"
# export LLM_MODEL="openrouter/deepseek/deepseek-v4-flash"          # per-file generator
# export SIFT_REVIEW_MODEL="openrouter/deepseek/deepseek-v4-pro"    # critic / holistic

export LLM_MODEL="openrouter/anthropic/claude-haiku-4.5"          # per-file generator
export SIFT_REVIEW_MODEL="openrouter/anthropic/claude-sonnet-5"    # critic / holistic

export SIFT_REVIEW_MODEL_BASE_URL="https://openrouter.ai/api/v1"
export SIFT_REVIEW_MODEL_KEY="${OPENROUTER_API_KEY}"
export DATABASE_URL=postgresql://sahil:sahil@localhost:5432/sift-benchmark
export SIFT_API_KEY="${SIFT_API_KEY:?source .env first so SIFT_API_KEY is set}"   # Bearer auth for POST /review

# Peak mode — assert on regardless of .env state.
export SIFT_REVIEW_EFFORT="high"
export CODEQL_ENABLED="1"
export VECTOR_DB_ENABLED="1"
export LOG_LEVEL="DEBUG"

# Embeddings: nomic-embed-text @ 768 via local Ollama (must be running).
export EMBEDDING_MODEL="ollama/nomic-embed-text"
export EMBEDDING_API_BASE="http://localhost:11434"
export EMBEDDING_DIMENSION="768"

echo "[bench_env] LLM=$LLM_MODEL critic=$SIFT_REVIEW_MODEL effort=$SIFT_REVIEW_EFFORT codeql=$CODEQL_ENABLED vector=$VECTOR_DB_ENABLED embed=$EMBEDDING_MODEL@$EMBEDDING_DIMENSION"
