"""Tests for the RAG pipeline."""

from __future__ import annotations

import pytest

from src.rag.chunker import chunk_documents, chunk_text
from src.rag.store import VectorStore, reset_client


class TestChunker:
    def test_empty_text(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text(self) -> None:
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunking_with_overlap(self) -> None:
        text = "First sentence. " * 50
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 200  # boundary search can extend slightly

    def test_chunk_documents(self) -> None:
        docs = [
            {"content": "Short doc.", "url": "https://a.com", "title": "A"},
            {"content": "Another document. " * 30, "url": "https://b.com", "title": "B"},
        ]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 2
        assert all("url" in c for c in chunks)
        assert chunks[0]["url"] == "https://a.com"


class TestVectorStore:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        reset_client()

    def test_add_and_query(self) -> None:
        store = VectorStore("test_collection")
        docs = [
            "Quantum computing uses qubits for parallel processing",
            "Classical computers use transistors and binary logic",
            "Machine learning requires large datasets for training",
        ]
        store.add_documents(docs)
        assert store.count == 3

        results = store.query("quantum qubits", n_results=2)
        assert len(results) == 2
        assert "quantum" in results[0]["content"].lower()

    def test_empty_store_query(self) -> None:
        store = VectorStore("empty_test")
        results = store.query("anything")
        assert results == []

    def test_add_with_metadata(self) -> None:
        store = VectorStore("meta_test")
        store.add_documents(
            ["Test document content"],
            metadatas=[{"url": "https://example.com", "title": "Test"}],
        )
        results = store.query("test", n_results=1)
        assert len(results) == 1
        assert results[0]["metadata"]["url"] == "https://example.com"

    def test_delete_collection(self) -> None:
        store = VectorStore("delete_test")
        store.add_documents(["Some content"])
        store.delete_collection()
        # Recreate to verify it was deleted
        new_store = VectorStore("delete_test")
        assert new_store.count == 0


class TestHybridRetriever:
    def test_hybrid_search(self) -> None:
        from src.rag.retriever import HybridRetriever

        reset_client()
        store = VectorStore("hybrid_test")

        docs = [
            "Quantum computing leverages quantum mechanical phenomena",
            "Machine learning algorithms learn from data patterns",
            "Blockchain provides decentralized transaction ledgers",
        ]
        metas = [
            {"url": "https://a.com"},
            {"url": "https://b.com"},
            {"url": "https://c.com"},
        ]

        retriever = HybridRetriever(store, documents=docs, metadatas=metas)
        retriever.add_documents(docs, metadatas=metas)

        results = retriever.retrieve("quantum computing", n_results=2)
        assert len(results) >= 1
        assert "quantum" in results[0]["content"].lower()

    def test_empty_retriever(self) -> None:
        from src.rag.retriever import HybridRetriever

        reset_client()
        store = VectorStore("empty_hybrid")
        retriever = HybridRetriever(store)

        results = retriever.retrieve("anything")
        assert results == []
