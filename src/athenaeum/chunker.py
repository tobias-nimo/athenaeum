"""Markdown chunking via LangChain RecursiveCharacterTextSplitter."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from athenaeum.models import ChunkMetadata

_DEFAULT_CHUNK_SIZE = 1500
_DEFAULT_CHUNK_OVERLAP = 200


@runtime_checkable
class TextSplitter(Protocol):
    """Minimal protocol for LangChain-compatible text splitters."""

    def split_text(self, text: str) -> list[str]: ...


def make_splitter(
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> TextSplitter:
    """Create a RecursiveCharacterTextSplitter for markdown.

    Args:
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.
        separators: Custom separator list. When ``None``, uses the built-in
            markdown-aware separators from ``Language.MARKDOWN``.

    Returns:
        A text splitter compatible with the ``TextSplitter`` protocol.
    """
    if separators is not None:
        return RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def auto_chunk_splitter(text: str) -> TextSplitter:
    """Choose chunk parameters automatically based on document length.

    Strategy:
    - **Short** (<5 000 chars): ``chunk_size=500``, ``chunk_overlap=50``.
      Small documents benefit from tight, granular chunks.
    - **Medium** (5 000–50 000 chars): ``chunk_size=1 500``, ``chunk_overlap=200``.
      Balanced defaults suitable for most documents.
    - **Large** (>50 000 chars): ``chunk_size=3 000``, ``chunk_overlap=400``.
      Wider context windows reduce fragmentation in very long documents.

    Args:
        text: Full document text used to decide chunk parameters.

    Returns:
        A configured ``TextSplitter``.
    """
    length = len(text)
    if length < 5_000:
        chunk_size, chunk_overlap = 500, 50
    elif length < 50_000:
        chunk_size, chunk_overlap = 1_500, 200
    else:
        chunk_size, chunk_overlap = 3_000, 400
    return make_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_markdown(
    markdown: str,
    doc_id: str,
    text_splitter: TextSplitter | None = None,
) -> list[ChunkMetadata]:
    """Split markdown into overlapping chunks with accurate 1-indexed line numbers.

    Args:
        markdown: Full markdown text.
        doc_id: Parent document ID.
        text_splitter: Optional splitter implementing ``split_text()``. Defaults to a
            markdown-aware ``RecursiveCharacterTextSplitter`` with
            ``chunk_size=1 500`` and ``chunk_overlap=200``.

    Returns:
        List of ``ChunkMetadata`` with accurate 1-indexed line numbers.
    """
    if not markdown:
        return []

    splitter = text_splitter or make_splitter()
    chunk_texts = splitter.split_text(markdown)

    chunks: list[ChunkMetadata] = []
    search_start = 0
    for i, chunk_text in enumerate(chunk_texts):
        start_char = markdown.find(chunk_text, search_start)
        if start_char == -1:
            start_char = search_start
        start_line = markdown[:start_char].count("\n") + 1
        end_line = start_line + chunk_text.count("\n")
        chunks.append(
            ChunkMetadata(
                doc_id=doc_id,
                chunk_index=i,
                start_line=start_line,
                end_line=end_line,
                text=chunk_text,
            )
        )
        search_start = start_char + 1

    return chunks
