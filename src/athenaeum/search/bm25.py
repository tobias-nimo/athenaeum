"""BM25 keyword search index."""

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from athenaeum.models import ChunkMetadata


@dataclass
class _Entry:
    chunk: ChunkMetadata
    tokens: list[str]


class BM25Index:
    """In-memory BM25 index over document chunks."""

    def __init__(self) -> None:
        self._entries: list[_Entry] = []
        self._bm25: BM25Okapi | None = None

    def _rebuild(self) -> None:
        if self._entries:
            corpus = [e.tokens for e in self._entries]
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def add_chunks(self, chunks: list[ChunkMetadata]) -> None:
        """Add chunks to the index and rebuild."""
        for chunk in chunks:
            self._entries.append(_Entry(chunk=chunk, tokens=self._tokenize(chunk.text)))
        self._rebuild()

    def remove_document(self, doc_id: str) -> None:
        """Remove all chunks for a document and rebuild."""
        self._entries = [e for e in self._entries if e.chunk.doc_id != doc_id]
        self._rebuild()

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_id: str | None = None,
        doc_ids: set[str] | None = None,
    ) -> list[tuple[ChunkMetadata, float]]:
        """Search the index, returning (chunk, score) pairs sorted by score descending."""
        if not self._bm25 or not self._entries:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        results: list[tuple[ChunkMetadata, float]] = []
        for i, score in enumerate(scores):
            entry = self._entries[i]
            if doc_id is not None and entry.chunk.doc_id != doc_id:
                continue
            if doc_ids is not None and entry.chunk.doc_id not in doc_ids:
                continue
            results.append((entry.chunk, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @property
    def size(self) -> int:
        return len(self._entries)
