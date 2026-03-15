"""Embeddings client for code similarity search (LiteLLM: Ollama, OpenAI, Gemini, etc.)."""
import logging
from typing import List

from litellm import aembedding

from src import config

logger = logging.getLogger(__name__)


async def get_embedding(text: str) -> List[float]:
    """Embed a single text string via the configured embedding provider."""
    results = await get_embeddings([text])
    return results[0]


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts via LiteLLM (configured provider).

    Returns one embedding vector per input text in the same order.
    """
    if not texts:
        logger.debug("[Embed] no texts, returning []")
        return []

    response = await aembedding(
        model=config.EMBEDDING_MODEL,
        input=texts,
        api_base=config.EMBEDDING_API_BASE or None,
        timeout=120.0,
    )
    out = [item["embedding"] for item in response.data]
    logger.debug("[Embed] done: %d embedding(s) with model %s", len(out), config.EMBEDDING_MODEL)
    return out


__all__ = ["get_embedding", "get_embeddings"]
