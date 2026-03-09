"""Smart text chunking with overlap and semantic boundary awareness."""

from __future__ import annotations

import re

from src.config import get_settings


def _find_boundary(text: str, pos: int, window: int = 100) -> int:
    """Find the nearest sentence or paragraph boundary near pos."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    segment = text[start:end]

    boundaries = [
        m.end() + start
        for m in re.finditer(r'[.!?]\s+|\n\n|\n(?=[A-Z#\-*])', segment)
    ]

    if boundaries:
        return min(boundaries, key=lambda b: abs(b - pos))
    return pos


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks respecting sentence boundaries.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum characters per chunk (default from settings).
        chunk_overlap: Overlap between consecutive chunks (default from settings).

    Returns:
        List of text chunks.
    """
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    if not text or not text.strip():
        return []

    text = text.strip()

    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + size

        if end < len(text):
            end = _find_boundary(text, end)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


def chunk_documents(
    documents: list[dict[str, str]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, str]]:
    """Chunk multiple documents, preserving metadata.

    Args:
        documents: List of dicts with 'content' and optional 'url', 'title'.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of chunk dicts with 'content', 'url', 'title', 'chunk_index'.
    """
    all_chunks: list[dict[str, str]] = []

    for doc in documents:
        content = doc.get("content", "")
        chunks = chunk_text(content, chunk_size, chunk_overlap)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "content": chunk,
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
                "chunk_index": str(i),
            })

    return all_chunks
