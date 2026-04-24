"""CHRONOS embedding layer — Google text-embedding-004 with batch support."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import settings

logger = logging.getLogger(__name__)

# Singleton embedding model
_embed_model: GoogleGenerativeAIEmbeddings | None = None

# Batch limits for Google embedding API
EMBEDDING_BATCH_SIZE = 50
EMBEDDING_CONCURRENCY = 3  # Max concurrent batch requests


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Get or create the singleton embedding model."""
    global _embed_model
    if _embed_model is None:
        _embed_model = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
    return _embed_model


async def embed_text(text: str) -> list[float]:
    """Embed a single text string. Returns a 768-dim vector."""
    model = get_embedding_model()
    return await model.aembed_query(text)


# Alias for retrieval.py import compatibility.
embed_query = embed_text


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts using batched API calls.

    Google's embedding API supports batch sizes up to ~100 texts per call.
    We batch in chunks of EMBEDDING_BATCH_SIZE and run up to
    EMBEDDING_CONCURRENCY batches concurrently for throughput.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors, one per input text, in order.
    """
    if not texts:
        return []

    if len(texts) == 1:
        vec = await embed_text(texts[0])
        return [vec]

    model = get_embedding_model()

    # Split into batches
    batches = [
        texts[i : i + EMBEDDING_BATCH_SIZE]
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE)
    ]

    logger.info(
        f"[embed] Embedding {len(texts)} texts in {len(batches)} batch(es) "
        f"(batch_size={EMBEDDING_BATCH_SIZE})"
    )

    sem = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

    async def _embed_batch(batch: list[str]) -> list[list[float]]:
        async with sem:
            try:
                return await model.aembed_documents(batch)
            except Exception as e:
                logger.error(f"[embed] Batch embedding failed ({len(batch)} texts): {e}")
                # Fallback: embed one at a time
                results = []
                for text in batch:
                    try:
                        vec = await model.aembed_query(text)
                        results.append(vec)
                    except Exception as inner_e:
                        logger.error(f"[embed] Single text embed failed: {inner_e}")
                        # Return zero vector as placeholder — will get low similarity scores
                        results.append([0.0] * settings.embedding_dimensions)
                return results

    # Run all batches with concurrency limit
    batch_results = await asyncio.gather(*[_embed_batch(b) for b in batches])

    # Flatten results preserving order
    all_embeddings: list[list[float]] = []
    for batch_result in batch_results:
        all_embeddings.extend(batch_result)

    assert len(all_embeddings) == len(texts), (
        f"Embedding count mismatch: got {len(all_embeddings)}, expected {len(texts)}"
    )

    return all_embeddings
