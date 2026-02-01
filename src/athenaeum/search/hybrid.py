"""Reciprocal Rank Fusion for combining ranked search results."""

from __future__ import annotations

from athenaeum.models import ChunkMetadata


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[ChunkMetadata, float]]],
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[ChunkMetadata, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for a chunk = sum over lists of 1 / (k + rank), where rank is
    1-indexed. Higher scores indicate better combined relevance.

    Args:
        ranked_lists: List of ranked result lists, each containing (chunk, score) pairs.
        k: RRF constant (default 60).
        top_k: Number of results to return.

    Returns:
        Merged and re-ranked list of (chunk, rrf_score) pairs.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, ChunkMetadata] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked_list, start=1):
            key = f"{chunk.doc_id}:{chunk.chunk_index}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            chunks[key] = chunk

    merged = [(chunks[key], score) for key, score in scores.items()]
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]
