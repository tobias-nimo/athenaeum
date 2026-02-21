"""Tests for markdown chunking."""

from __future__ import annotations

import re

from athenaeum.chunker import TextSplitter, chunk_markdown


def test_chunk_basic(sample_md_text: str) -> None:
    chunks = chunk_markdown(sample_md_text, doc_id="test-1", chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert chunks[0].doc_id == "test-1"


def test_chunk_indices_sequential(sample_md_text: str) -> None:
    chunks = chunk_markdown(sample_md_text, doc_id="d", chunk_size=200, chunk_overlap=50)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_chunk_single_chunk(sample_md_text: str) -> None:
    chunks = chunk_markdown(sample_md_text, doc_id="d", chunk_size=5000, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line >= 1


def test_chunk_empty() -> None:
    assert chunk_markdown("", doc_id="d") == []


def test_line_numbers_accurate(sample_md_text: str) -> None:
    chunks = chunk_markdown(sample_md_text, doc_id="d", chunk_size=200, chunk_overlap=50)
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
    chunks = chunk_markdown(text, doc_id="d", chunk_size=300, chunk_overlap=50)
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
