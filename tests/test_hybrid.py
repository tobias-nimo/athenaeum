"""Tests for reciprocal rank fusion."""

from athenaeum.models import ChunkMetadata
from athenaeum.search.hybrid import reciprocal_rank_fusion


def _chunk(doc_id: str, idx: int) -> ChunkMetadata:
    return ChunkMetadata(
        doc_id=doc_id, chunk_index=idx, start_line=1, end_line=10, text=f"chunk {idx}"
    )


def test_rrf_single_list() -> None:
    ranked = [(_chunk("d1", 0), 0.9), (_chunk("d1", 1), 0.5)]
    results = reciprocal_rank_fusion([ranked], k=60, top_k=10)
    assert len(results) == 2
    # First item should have higher RRF score
    assert results[0][1] > results[1][1]


def test_rrf_two_lists_boost_overlap() -> None:
    # chunk 0 appears in both lists -> should be boosted
    list1 = [(_chunk("d1", 0), 0.9), (_chunk("d1", 1), 0.5)]
    list2 = [(_chunk("d1", 2), 0.8), (_chunk("d1", 0), 0.7)]
    results = reciprocal_rank_fusion([list1, list2], k=60, top_k=10)
    # chunk 0 should be first due to appearing in both lists
    assert results[0][0].chunk_index == 0


def test_rrf_top_k() -> None:
    ranked = [(_chunk("d1", i), 1.0 / (i + 1)) for i in range(20)]
    results = reciprocal_rank_fusion([ranked], k=60, top_k=5)
    assert len(results) == 5


def test_rrf_empty() -> None:
    assert reciprocal_rank_fusion([], top_k=10) == []
    assert reciprocal_rank_fusion([[]], top_k=10) == []
