"""Embedding model wrapper using sentence-transformers."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.config import get_settings

logger = logging.getLogger(__name__)

# Lazy-loaded singleton to avoid slow import on startup
_model: object | None = None


def _get_model() -> object:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        _model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
        logger.info("Loaded embedding model: %s", settings.embedding_model)
    return _model


def embed_texts(texts: Sequence[str]) -> NDArray[np.float32]:
    """Embed a batch of texts into dense vectors.

    Args:
        texts: Texts to embed.

    Returns:
        Array of shape (len(texts), embedding_dim).
    """
    model = _get_model()
    embeddings: NDArray[np.float32] = model.encode(  # type: ignore[attr-defined]
        list(texts),
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings


def embed_query(query: str) -> NDArray[np.float32]:
    """Embed a single query string.

    Args:
        query: The query text.

    Returns:
        1D array of shape (embedding_dim,).
    """
    result = embed_texts([query])
    return result[0]  # type: ignore[no-any-return]


def reset_model() -> None:
    """Reset cached model (for testing)."""
    global _model
    _model = None
