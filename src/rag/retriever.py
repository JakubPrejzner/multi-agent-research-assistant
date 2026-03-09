"""Hybrid retriever combining dense (cosine) and sparse (BM25) search."""

from __future__ import annotations

import logging
from typing import Any

from rank_bm25 import BM25Okapi

from src.rag.store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines dense vector search with BM25 sparse retrieval."""

    def __init__(
        self,
        store: VectorStore,
        documents: list[str] | None = None,
        metadatas: list[dict[str, str]] | None = None,
        dense_weight: float = 0.6,
    ) -> None:
        self._store = store
        self._dense_weight = dense_weight
        self._sparse_weight = 1.0 - dense_weight

        self._documents = documents or []
        self._metadatas = metadatas or []
        self._bm25: BM25Okapi | None = None

        if self._documents:
            self._build_bm25()

    def _build_bm25(self) -> None:
        """Build BM25 index from documents."""
        tokenized = [doc.lower().split() for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, str]] | None = None,
    ) -> None:
        """Add documents to both dense and sparse indexes."""
        if not documents:
            return

        metas = metadatas or [{} for _ in documents]

        self._store.add_documents(documents, metadatas=metas)
        self._documents.extend(documents)
        self._metadatas.extend(metas)
        self._build_bm25()

    def _bm25_search(self, query: str, n_results: int) -> list[dict[str, Any]]:
        """Sparse retrieval via BM25."""
        if not self._bm25 or not self._documents:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:n_results]

        max_score = max(scores) if max(scores) > 0 else 1.0

        results: list[dict[str, Any]] = []
        for idx in scored_indices:
            if scores[idx] > 0:
                results.append({
                    "content": self._documents[idx],
                    "metadata": self._metadatas[idx] if idx < len(self._metadatas) else {},
                    "score": float(scores[idx] / max_score),
                })
        return results

    def retrieve(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Hybrid retrieval combining dense and sparse results.

        Args:
            query: The search query.
            n_results: Number of results to return.

        Returns:
            Ranked list of dicts with 'content', 'metadata', 'score'.
        """
        dense_results = self._store.query(query, n_results=n_results * 2)
        sparse_results = self._bm25_search(query, n_results * 2)

        scored: dict[str, dict[str, Any]] = {}

        for r in dense_results:
            key = r["content"][:100]
            dense_score = 1.0 - float(r.get("distance", 0.5))
            scored[key] = {
                "content": r["content"],
                "metadata": r.get("metadata", {}),
                "score": dense_score * self._dense_weight,
            }

        for r in sparse_results:
            key = r["content"][:100]
            sparse_score = float(r.get("score", 0.0))
            if key in scored:
                scored[key]["score"] += sparse_score * self._sparse_weight
            else:
                scored[key] = {
                    "content": r["content"],
                    "metadata": r.get("metadata", {}),
                    "score": sparse_score * self._sparse_weight,
                }

        ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:n_results]
