"""Tests for Vector search index."""

from __future__ import annotations

import uuid

from langchain_core.embeddings import Embeddings

from athenaeum.models import ChunkMetadata
from athenaeum.search.vector import VectorIndex


class FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        # Simple hash-based embedding to 32 dims
        vec = [0.0] * 32
        for i, ch in enumerate(text.encode()):
            vec[i % 32] += ch / 256.0
        # Normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


def _make_index() -> VectorIndex:
    return VectorIndex(embeddings=FakeEmbeddings(), collection_name=f"test-{uuid.uuid4().hex}")


def _make_chunks() -> list[ChunkMetadata]:
    return [
        ChunkMetadata(
            doc_id="d1", chunk_index=0, start_line=1, end_line=10,
            text="python programming language tutorial",
        ),
        ChunkMetadata(
            doc_id="d1", chunk_index=1, start_line=11, end_line=20,
            text="java programming enterprise applications",
        ),
        ChunkMetadata(
            doc_id="d2", chunk_index=0, start_line=1, end_line=10,
            text="python data science machine learning",
        ),
    ]


def test_vector_add_and_search() -> None:
    idx = _make_index()
    idx.add_chunks(_make_chunks())
    results = idx.search("python programming", top_k=3)
    assert len(results) > 0
    assert all(isinstance(c, ChunkMetadata) for c, _ in results)


def test_vector_search_empty() -> None:
    idx = _make_index()
    results = idx.search("anything")
    assert results == []


def test_vector_filter_by_doc_id() -> None:
    idx = _make_index()
    idx.add_chunks(_make_chunks())
    results = idx.search("python", doc_id="d2")
    assert all(c.doc_id == "d2" for c, _ in results)


def test_vector_remove_document() -> None:
    idx = _make_index()
    idx.add_chunks(_make_chunks())
    idx.remove_document("d1")
    results = idx.search("python", top_k=10)
    assert all(c.doc_id == "d2" for c, _ in results)


def test_vector_add_empty() -> None:
    idx = _make_index()
    idx.add_chunks([])  # should not raise
