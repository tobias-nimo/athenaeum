"""Main orchestrator class for Athenaeum."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Literal

from langchain_core.embeddings import Embeddings

from athenaeum.chunker import chunk_markdown
from athenaeum.config import AthenaeumConfig
from athenaeum.document_store import DocumentStore
from athenaeum.models import ChunkMetadata, ContentSearchHit, Document, Excerpt, SearchHit
from athenaeum.ocr import OCRProvider, get_ocr_provider
from athenaeum.search.bm25 import BM25Index
from athenaeum.search.hybrid import reciprocal_rank_fusion
from athenaeum.search.vector import VectorIndex
from athenaeum.storage import StorageManager
from athenaeum.toc import extract_toc


class Athenaeum:
    """Main entry point for the Athenaeum knowledge base."""

    def __init__(
        self,
        embeddings: Embeddings,
        config: AthenaeumConfig | None = None,
        ocr_provider: OCRProvider | None = None,
    ) -> None:
        self._config = config or AthenaeumConfig()
        self._storage = StorageManager(self._config.storage_dir)
        self._doc_store = DocumentStore(self._storage)
        self._ocr = ocr_provider or get_ocr_provider("markitdown")
        self._bm25 = BM25Index()
        self._vector = VectorIndex(
            embeddings=embeddings,
            persist_directory=self._storage.ensure_chroma_dir(),
            collection_name="athenaeum",
        )
        self._reindex_bm25()

    def _reindex_bm25(self) -> None:
        """Rebuild BM25 index from all stored documents."""
        for doc in self._doc_store.list_all():
            md_path = Path(doc.path_to_md)
            if md_path.exists():
                text = md_path.read_text()
                chunks = chunk_markdown(
                    text, doc.id, self._config.chunk_size, self._config.chunk_overlap
                )
                self._bm25.add_chunks(chunks)

    def load_doc(self, path: str) -> str:
        """Load a document into the knowledge base.

        Args:
            path: Path to the document file.

        Returns:
            The document ID.
        """
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        supported = self._ocr.supported_extensions()
        if ".*" not in supported and ext not in supported:
            raise ValueError(
                f"Unsupported file type: {ext}. Supported: {sorted(supported)}"
            )

        doc_id = uuid.uuid4().hex[:12]

        # Copy raw file
        raw_dest = self._storage.raw_path(doc_id, ext)
        shutil.copy2(file_path, raw_dest)

        # Convert to markdown
        markdown = self._ocr.convert(file_path)
        md_dest = self._storage.content_md_path(doc_id)
        md_dest.write_text(markdown)

        # Extract metadata
        toc = extract_toc(markdown)
        lines = markdown.split("\n")

        doc = Document(
            id=doc_id,
            name=file_path.name,
            path_to_raw=str(raw_dest),
            path_to_md=str(md_dest),
            num_lines=len(lines),
            table_of_contents=toc,
            file_size=file_path.stat().st_size,
            file_type=ext,
        )
        self._doc_store.add(doc)

        # Index
        chunks = chunk_markdown(
            markdown, doc_id, self._config.chunk_size, self._config.chunk_overlap
        )
        self._bm25.add_chunks(chunks)
        self._vector.add_chunks(chunks)

        return doc_id

    def list_docs(self) -> list[SearchHit]:
        """List all documents in the knowledge base."""
        results = []
        for doc in self._doc_store.list_all():
            results.append(
                SearchHit(
                    id=doc.id,
                    name=doc.name,
                    num_lines=doc.num_lines,
                    table_of_contents=doc.format_toc(),
                )
            )
        return results

    def search_docs(
        self,
        query: str,
        top_k: int = 10,
        scope: Literal["names", "contents"] = "contents",
        strategy: Literal["hybrid", "bm25", "vector"] = "hybrid",
    ) -> list[SearchHit]:
        """Search across all documents.

        Args:
            query: Search query text.
            top_k: Maximum number of results.
            scope: ``"contents"`` to search within documents, ``"names"`` to search names only.
            strategy: Search strategy (only for ``scope="contents"``).

        Returns:
            Ranked list of matching documents.
        """
        if scope == "names":
            return self._search_by_name(query, top_k)

        chunks = self._search_chunks(query, top_k=top_k * 3, strategy=strategy)

        # Aggregate chunks by document
        doc_scores: dict[str, float] = {}
        doc_snippets: dict[str, str] = {}
        for chunk, score in chunks:
            if chunk.doc_id not in doc_scores or score > doc_scores[chunk.doc_id]:
                doc_scores[chunk.doc_id] = score
                doc_snippets[chunk.doc_id] = chunk.text[:200]

        results = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            doc = self._doc_store.get(doc_id)
            if doc is None:
                continue
            results.append(
                SearchHit(
                    id=doc.id,
                    name=doc.name,
                    num_lines=doc.num_lines,
                    table_of_contents=doc.format_toc(),
                    score=score,
                    snippet=doc_snippets.get(doc_id, ""),
                )
            )

        return results[:top_k]

    def search_doc_contents(
        self,
        doc_id: str,
        query: str,
        top_k: int = 5,
        strategy: Literal["hybrid", "bm25", "vector"] = "hybrid",
    ) -> list[ContentSearchHit]:
        """Search within a specific document.

        Args:
            doc_id: Document identifier.
            query: Search query text.
            top_k: Maximum number of results.
            strategy: Search strategy.

        Returns:
            List of matching content fragments.
        """
        doc = self._doc_store.get(doc_id)
        if doc is None:
            raise ValueError(f"Document not found: {doc_id}")

        chunks = self._search_chunks(query, top_k=top_k, strategy=strategy, doc_id=doc_id)

        return [
            ContentSearchHit(
                doc_id=chunk.doc_id,
                line_range=(chunk.start_line, chunk.end_line),
                text=chunk.text,
                score=score,
            )
            for chunk, score in chunks
        ]

    def read_doc(
        self,
        doc_id: str,
        start_line: int = 1,
        end_line: int = 100,
    ) -> Excerpt:
        """Read a range of lines from a document.

        Args:
            doc_id: Document identifier.
            start_line: Starting line number (1-indexed).
            end_line: Ending line number (1-indexed, inclusive).

        Returns:
            An ``Excerpt`` with the requested lines.
        """
        doc = self._doc_store.get(doc_id)
        if doc is None:
            raise ValueError(f"Document not found: {doc_id}")

        md_path = Path(doc.path_to_md)
        lines = md_path.read_text().split("\n")

        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        selected = lines[start_idx:end_idx]

        return Excerpt(
            doc_id=doc_id,
            line_range=(start_idx + 1, end_idx),
            text="\n".join(selected),
            total_lines=len(lines),
        )

    def _search_by_name(self, query: str, top_k: int) -> list[SearchHit]:
        query_lower = query.lower()
        scored = []
        for doc in self._doc_store.list_all():
            name_lower = doc.name.lower()
            if query_lower in name_lower:
                # Simple relevance: exact match > contains
                score = 1.0 if query_lower == name_lower else 0.5
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchHit(
                id=doc.id,
                name=doc.name,
                num_lines=doc.num_lines,
                table_of_contents=doc.format_toc(),
                score=score,
            )
            for doc, score in scored[:top_k]
        ]

    def _search_chunks(
        self,
        query: str,
        top_k: int,
        strategy: Literal["hybrid", "bm25", "vector"] = "hybrid",
        doc_id: str | None = None,
    ) -> list[tuple[ChunkMetadata, float]]:
        """Internal: run search using the given strategy."""
        if strategy == "bm25":
            return self._bm25.search(query, top_k=top_k, doc_id=doc_id)

        if strategy == "vector":
            return self._vector.search(query, top_k=top_k, doc_id=doc_id)

        # hybrid
        bm25_results = self._bm25.search(query, top_k=top_k, doc_id=doc_id)
        vector_results = self._vector.search(query, top_k=top_k, doc_id=doc_id)
        return reciprocal_rank_fusion(
            [bm25_results, vector_results],
            k=self._config.rrf_k,
            top_k=top_k,
        )
