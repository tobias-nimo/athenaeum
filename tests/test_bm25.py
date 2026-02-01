"""Tests for BM25 search index."""

from athenaeum.models import ChunkMetadata
from athenaeum.search.bm25 import BM25Index


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


def test_bm25_search_basic() -> None:
    idx = BM25Index()
    idx.add_chunks(_make_chunks())
    results = idx.search("python programming", top_k=3)
    assert len(results) > 0
    # The python chunks should rank higher
    assert results[0][0].text.startswith("python")


def test_bm25_search_empty() -> None:
    idx = BM25Index()
    assert idx.search("anything") == []


def test_bm25_filter_by_doc_id() -> None:
    idx = BM25Index()
    idx.add_chunks(_make_chunks())
    results = idx.search("python", doc_id="d2")
    assert all(c.doc_id == "d2" for c, _ in results)


def test_bm25_remove_document() -> None:
    idx = BM25Index()
    idx.add_chunks(_make_chunks())
    assert idx.size == 3
    idx.remove_document("d1")
    assert idx.size == 1
    results = idx.search("python")
    assert all(c.doc_id == "d2" for c, _ in results)


def test_bm25_top_k() -> None:
    idx = BM25Index()
    idx.add_chunks(_make_chunks())
    results = idx.search("python", top_k=1)
    assert len(results) == 1
