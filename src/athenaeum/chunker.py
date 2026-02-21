"""Markdown chunking via LangChain RecursiveCharacterTextSplitter."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from athenaeum.models import ChunkMetadata


@runtime_checkable
class TextSplitter(Protocol):
    """Minimal protocol for LangChain-compatible text splitters."""

    def split_text(self, text: str) -> list[str]: ...


def _default_splitter(chunk_size: int, chunk_overlap: int) -> TextSplitter:
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def chunk_markdown(
    markdown: str,
    doc_id: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    text_splitter: TextSplitter | None = None,
) -> list[ChunkMetadata]:
    """Split markdown into overlapping chunks with accurate 1-indexed line numbers.

    Args:
        markdown: Full markdown text.
        doc_id: Parent document ID.
        chunk_size: Characters per chunk (ignored when text_splitter is provided).
        chunk_overlap: Overlap in characters (ignored when text_splitter is provided).
        text_splitter: Optional custom splitter implementing split_text(). Defaults to
            a markdown-aware RecursiveCharacterTextSplitter that respects heading boundaries.

    Returns:
        List of ChunkMetadata with accurate 1-indexed line numbers.
    """
    if not markdown:
        return []

    splitter = text_splitter or _default_splitter(chunk_size, chunk_overlap)
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
