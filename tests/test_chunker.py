"""Tests for markdown chunking."""

from athenaeum.chunker import chunk_markdown


def test_chunk_basic(sample_md_text: str) -> None:
    chunks = chunk_markdown(sample_md_text, doc_id="test-1", chunk_size=10, chunk_overlap=3)
    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert chunks[0].doc_id == "test-1"


def test_chunk_indices_sequential(sample_md_text: str) -> None:
    chunks = chunk_markdown(sample_md_text, doc_id="d", chunk_size=10, chunk_overlap=2)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_chunk_coverage(sample_md_text: str) -> None:
    """All lines should be covered by at least one chunk."""
    total_lines = len(sample_md_text.split("\n"))
    chunks = chunk_markdown(sample_md_text, doc_id="d", chunk_size=10, chunk_overlap=2)
    covered = set()
    for c in chunks:
        for line in range(c.start_line, c.end_line + 1):
            covered.add(line)
    assert covered == set(range(1, total_lines + 1))


def test_chunk_single_chunk() -> None:
    text = "line 1\nline 2\nline 3"
    chunks = chunk_markdown(text, doc_id="d", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3


def test_chunk_empty() -> None:
    assert chunk_markdown("", doc_id="d") == []


def test_chunk_text_content() -> None:
    text = "alpha\nbeta\ngamma"
    chunks = chunk_markdown(text, doc_id="d", chunk_size=2, chunk_overlap=0)
    assert chunks[0].text == "alpha\nbeta"
