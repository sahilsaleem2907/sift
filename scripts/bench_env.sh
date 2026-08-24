#!/usr/bin/env bash
# Benchmark "peak" config overlay. Source AFTER .env:
#   set -a; source .env; source scripts/bench_env.sh; set +a
# Overrides only the LLM models + asserts peak toggles; everything else (DB, secrets,
# semgrep/linter flags) is inherited from .env. config.py's load_dotenv(override=False)
# means these shell exports win over any .env value.

# Direct Anthropic — Opus generates per-file (recall-critical primary/agentic pass), Sonnet
# critiques/holistic (precision gate, fired per-finding at high effort). Direct Anthropic (NOT
# via OpenRouter) is required for two reasons: (1) prompt caching — LiteLLM only honors
# cache_control content blocks on its Anthropic provider; (2) with both LLM_API_BASE and
# LLM_API_KEY set, _call_llm takes the raw-httpx path which has no caching and no token capture.
: "${ANTHROPIC_API_KEY:?source .env first so ANTHROPIC_API_KEY is set (LiteLLM reads it for anthropic/ models)}"
export LLM_MODEL="anthropic/claude-sonnet-4-6"          # per-file generator (primary / agentic)
export SIFT_REVIEW_MODEL="anthropic/claude-sonnet-4-6"  # critic / holistic (set explicitly so the
                                                        # full LLM critic path runs, not just dedupe)

# Neutralize any OpenRouter/Ollama base+key from .env so calls route through LiteLLM's Anthropic
# provider — not the OpenAI-compatible httpx path. This is load-bearing for caching.
# EXPORT EMPTY, do not `unset`: config.py runs load_dotenv(override=False) at import, which
# re-reads .env for any var that is absent from the environment — so `unset LLM_API_BASE` would
# be silently refilled from .env. An exported empty value is "present", so load_dotenv skips it,
# and config.py treats "" as None.
export LLM_API_BASE=""
export LLM_API_KEY=""
export SIFT_REVIEW_MODEL_BASE_URL=""
export SIFT_REVIEW_MODEL_KEY=""

# Legacy OpenRouter config (kept for reference; re-enable to route via OpenRouter — note this
# disables prompt caching + cost capture because it takes the httpx path):
#   export LLM_API_BASE="https://openrouter.ai/api/v1"
#   export LLM_API_KEY="${OPENROUTER_API_KEY}"
#   export LLM_MODEL="openrouter/anthropic/claude-opus-4"
#   export SIFT_REVIEW_MODEL="openrouter/anthropic/claude-sonnet-4.6"
#   export SIFT_REVIEW_MODEL_BASE_URL="https://openrouter.ai/api/v1"
#   export SIFT_REVIEW_MODEL_KEY="${OPENROUTER_API_KEY}"

export DATABASE_URL=postgresql://sahil:sahil@localhost:5432/sift-benchmark
export SIFT_API_KEY="${SIFT_API_KEY:?source .env first so SIFT_API_KEY is set}"   # Bearer auth for POST /review

# Peak mode — assert on regardless of .env state.
export SIFT_REVIEW_EFFORT="high"
export SIFT_COST_TRACKING="1"   # emit a per-review [cost] line (tokens/model, cache read/write, USD)
export CODEQL_ENABLED="1"
export CODEQL_SUITE="security-extended"
export CODEQL_TIMEOUT=1500
export VECTOR_DB_ENABLED="1"
export LOG_LEVEL="DEBUG"

# Embeddings: nomic-embed-text @ 768 via local Ollama (must be running).
export EMBEDDING_MODEL="ollama/nomic-embed-text"
export EMBEDDING_API_BASE="http://localhost:11434"
export EMBEDDING_DIMENSION="768"

echo "[bench_env] LLM=$LLM_MODEL critic=$SIFT_REVIEW_MODEL effort=$SIFT_REVIEW_EFFORT cost_tracking=$SIFT_COST_TRACKING codeql=$CODEQL_ENABLED/$CODEQL_SUITE vector=$VECTOR_DB_ENABLED embed=$EMBEDDING_MODEL@$EMBEDDING_DIMENSION"
