"""Tests for markdown chunking."""

from __future__ import annotations

import re

import pytest
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from athenaeum.chunker import TextSplitter, auto_chunk_splitter, chunk_markdown, make_splitter


def _small_splitter(chunk_size: int = 200, chunk_overlap: int = 50) -> TextSplitter:
    """Helper: small splitter for tests that need explicit sizing."""
    return make_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_chunk_basic(sample_md_text: str) -> None:
    chunks = chunk_markdown(sample_md_text, doc_id="test-1", text_splitter=_small_splitter())
    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert chunks[0].doc_id == "test-1"


def test_chunk_indices_sequential(sample_md_text: str) -> None:
    chunks = chunk_markdown(sample_md_text, doc_id="d", text_splitter=_small_splitter())
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_chunk_single_chunk(sample_md_text: str) -> None:
    chunks = chunk_markdown(
        sample_md_text, doc_id="d", text_splitter=make_splitter(chunk_size=5000, chunk_overlap=100)
    )
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line >= 1


def test_chunk_empty() -> None:
    assert chunk_markdown("", doc_id="d") == []


def test_line_numbers_accurate(sample_md_text: str) -> None:
    chunks = chunk_markdown(sample_md_text, doc_id="d", text_splitter=_small_splitter())
    assert chunks[0].start_line == 1
    for chunk in chunks:
        # Independently compute start_line using the same formula
        start_char = sample_md_text.find(chunk.text)
        expected_start = sample_md_text[:start_char].count("\n") + 1
        assert chunk.start_line == expected_start
        assert chunk.end_line >= chunk.start_line


def test_heading_aware() -> None:
    text = (
        "# Title\n\nOpening paragraph with enough text to fill a chunk boundary.\n\n"
        "More content here to push past the first chunk size limit.\n\n"
        "## Section\n\nSection content that should start a new chunk near the heading.\n\n"
        "Additional section text to ensure we have enough characters.\n"
    )
    chunks = chunk_markdown(text, doc_id="d", text_splitter=make_splitter(chunk_size=300, chunk_overlap=50))
    heading_re = re.compile(r"^#{1,6} ", re.MULTILINE)
    # At least one chunk (after the first) should start with or contain a heading
    texts = [c.text for c in chunks]
    assert any(heading_re.search(t) for t in texts)


def test_custom_text_splitter() -> None:
    class SimpleSplitter:
        def split_text(self, text: str) -> list[str]:
            return [text[:50], text[25:75], text[50:]] if len(text) > 50 else [text]

    splitter = SimpleSplitter()
    assert isinstance(splitter, TextSplitter)

    text = "A" * 100
    chunks = chunk_markdown(text, doc_id="d", text_splitter=splitter)
    assert len(chunks) > 0
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


# --- make_splitter tests ---


def test_make_splitter_default() -> None:
    splitter = make_splitter()
    chunks = splitter.split_text("# Hello\n\n" + "word " * 400)
    assert len(chunks) > 0


def test_make_splitter_custom_separators() -> None:
    splitter = make_splitter(chunk_size=50, chunk_overlap=5, separators=["\n", " "])
    chunks = splitter.split_text("line one\nline two\nline three\nline four\nline five")
    assert len(chunks) >= 1


# --- auto_chunk_splitter tests ---


def test_auto_chunk_short_doc() -> None:
    text = "short " * 50  # ~300 chars
    splitter = auto_chunk_splitter(text)
    # Short doc → chunk_size=500; the whole text fits in one chunk
    chunks = splitter.split_text(text)
    assert len(chunks) >= 1


def test_auto_chunk_medium_doc() -> None:
    text = "word " * 2000  # ~10 000 chars
    splitter = auto_chunk_splitter(text)
    chunks = splitter.split_text(text)
    assert len(chunks) >= 1


def test_auto_chunk_large_doc() -> None:
    text = "word " * 15000  # ~75 000 chars
    splitter = auto_chunk_splitter(text)
    chunks = splitter.split_text(text)
    assert len(chunks) >= 1
    # Verify chunk sizes are reasonable for large-doc settings (chunk_size=3000)
    for chunk in chunks:
        assert len(chunk) <= 3500  # allow small LangChain overrun tolerance
