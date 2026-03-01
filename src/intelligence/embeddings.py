"""Ollama embeddings client for code similarity search."""
import logging
from typing import List

import httpx

from src import config

logger = logging.getLogger(__name__)

_BATCH_SIZE = 20


async def get_embedding(text: str) -> List[float]:
    """Embed a single text string via Ollama."""
    results = await get_embeddings([text])
    return results[0]


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts via Ollama, batching if needed.

    Calls POST /api/embed with the configured EMBEDDING_MODEL.
    Returns one embedding vector per input text in the same order.
    """
    if not texts:
        logger.debug("[Embed] no texts, returning []")
        return []

    all_embeddings: List[List[float]] = []
    base_url = config.OLLAMA_BASE_URL
    model = config.EMBEDDING_MODEL
    num_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE
    logger.debug(
        "[Embed] start: %d text(s), model=%s, batch_size=%d, num_batches=%d",
        len(texts), model, _BATCH_SIZE, num_batches,
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            batch_idx = i // _BATCH_SIZE
            logger.debug("[Embed] batch %d/%d: %d text(s)", batch_idx + 1, num_batches, len(batch))
            payload = {"model": model, "input": batch}
            try:
                r = await client.post(f"{base_url}/api/embed", json=payload)
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPError as e:
                logger.error("Ollama embed request failed (batch %d): %s", i // _BATCH_SIZE, e)
                raise

            embeddings = data.get("embeddings")
            if not embeddings or len(embeddings) != len(batch):
                logger.warning(
                    "Unexpected embed response shape: expected %d embeddings, got %s",
                    len(batch),
                    len(embeddings) if embeddings else 0,
                )
                raise ValueError(f"Ollama /api/embed returned {len(embeddings) if embeddings else 0} embeddings for {len(batch)} inputs")

            all_embeddings.extend(embeddings)
            logger.debug("[Embed] batch %d/%d: got %d embedding(s)", batch_idx + 1, num_batches, len(embeddings))

    logger.debug("[Embed] done: %d total embedding(s) with model %s", len(all_embeddings), model)
    return all_embeddings


__all__ = ["get_embedding", "get_embeddings"]
