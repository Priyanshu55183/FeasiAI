"""
Embedding model wrapper.

Isolates the embedding model behind a clean interface so we can:
1. Swap models without touching retrieval logic.
2. Lazy-load the model (sentence-transformers is heavy on import).
3. Add caching or batching later without changing callers.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from feasiai import config

logger = logging.getLogger(__name__)

# Module-level singleton to avoid reloading the model
_model_instance = None


def get_embedding_model():
    """Lazy-load and cache the sentence-transformers model."""
    global _model_instance
    if _model_instance is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        _model_instance = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model_instance


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts and return their vector representations.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)

    # Convert numpy arrays to lists for serialization
    if isinstance(embeddings, np.ndarray):
        return embeddings.tolist()
    return [e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings]


def embed_query(query: str) -> list[float]:
    """Embed a single query text."""
    result = embed_texts([query])
    return result[0] if result else []
