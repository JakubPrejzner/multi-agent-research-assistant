"""ChromaDB vector store interface."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from src.config import get_settings
from src.rag.embeddings import embed_texts

logger = logging.getLogger(__name__)

_client: ClientAPI | None = None


def get_chroma_client() -> ClientAPI:
    """Return a ChromaDB client (persistent local or HTTP)."""
    global _client  # noqa: PLW0603
    if _client is not None:
        return _client

    settings = get_settings()
    if settings.chroma_use_local:
        _client = chromadb.Client()
        logger.info("Using ephemeral ChromaDB client")
    else:
        _client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        logger.info("Connected to ChromaDB at %s:%d", settings.chroma_host, settings.chroma_port)
    return _client


def reset_client() -> None:
    """Reset cached client (for testing)."""
    global _client  # noqa: PLW0603
    _client = None


class VectorStore:
    """Manages document embeddings in ChromaDB."""

    def __init__(self, collection_name: str, client: ClientAPI | None = None) -> None:
        self._client = client or get_chroma_client()
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_name = collection_name

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def count(self) -> int:
        return self._collection.count()

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, str]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Add documents to the vector store.

        Args:
            documents: Text content to embed and store.
            metadatas: Optional metadata for each document.
            ids: Optional custom IDs (auto-generated if not provided).
        """
        if not documents:
            return

        embeddings = embed_texts(documents).tolist()

        if ids is None:
            existing = self._collection.count()
            ids = [f"doc_{existing + i}" for i in range(len(documents))]

        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,  # type: ignore[arg-type]
            ids=ids,
        )
        logger.info("Added %d documents to collection '%s'", len(documents), self._collection_name)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Query the store for similar documents.

        Args:
            query_text: The query string.
            n_results: Number of results to return.

        Returns:
            List of dicts with 'content', 'metadata', 'distance'.
        """
        query_embedding = embed_texts([query_text]).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, max(1, self._collection.count())),
        )

        output: list[dict[str, Any]] = []
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        for doc, meta, dist in zip(
            documents[0], metadatas[0], distances[0]
        ):
            output.append({
                "content": doc,
                "metadata": meta or {},
                "distance": dist,
            })

        return output

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self._client.delete_collection(self._collection_name)
        logger.info("Deleted collection: %s", self._collection_name)
