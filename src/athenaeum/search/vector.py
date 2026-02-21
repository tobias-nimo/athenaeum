"""Vector similarity search index using LangChain + Chroma."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from athenaeum.models import ChunkMetadata


class VectorIndex:
    """Vector similarity search backed by Chroma."""

    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: str | Path | None = None,
        collection_name: str = "athenaeum",
    ) -> None:
        kwargs: dict[str, object] = {
            "embedding_function": embeddings,
            "collection_name": collection_name,
            "collection_metadata": {"hnsw:space": "cosine"},
        }
        if persist_directory is not None:
            kwargs["persist_directory"] = str(persist_directory)
        self._store = Chroma(**kwargs)  # type: ignore[arg-type]
        self._embeddings = embeddings

    def add_chunks(self, chunks: list[ChunkMetadata]) -> None:
        """Add chunks to the vector store."""
        if not chunks:
            return
        texts = [c.text for c in chunks]
        metadatas = [c.model_dump() for c in chunks]
        ids = [f"{c.doc_id}:{c.chunk_index}" for c in chunks]
        self._store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def remove_document(self, doc_id: str) -> None:
        """Remove all chunks for a document from the vector store."""
        self._store._collection.delete(where={"doc_id": doc_id})

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_id: str | None = None,
        doc_ids: set[str] | None = None,
        similarity_threshold: float | None = None,
    ) -> list[tuple[ChunkMetadata, float]]:
        """Search for similar chunks, returning (chunk, score) pairs.

        Scores are cosine similarity scores in [0, 1] (higher = more similar).
        """
        kwargs: dict[str, object] = {"k": top_k}
        if doc_id is not None:
            kwargs["filter"] = {"doc_id": doc_id}
        elif doc_ids is not None:
            kwargs["filter"] = {"doc_id": {"$in": sorted(doc_ids)}}

        results = self._store.similarity_search_with_relevance_scores(query, **kwargs)  # type: ignore[arg-type]

        output: list[tuple[ChunkMetadata, float]] = []
        for doc, score in results:
            if similarity_threshold is not None and score < similarity_threshold:
                continue
            meta = doc.metadata
            chunk = ChunkMetadata(
                doc_id=meta["doc_id"],
                chunk_index=meta["chunk_index"],
                start_line=meta["start_line"],
                end_line=meta["end_line"],
                text=meta["text"],
            )
            output.append((chunk, float(score)))

        return output
