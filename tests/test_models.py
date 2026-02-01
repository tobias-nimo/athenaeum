"""Tests for Athenaeum data models."""

from athenaeum.models import (
    ChunkMetadata,
    ContentSearchHit,
    Document,
    Excerpt,
    Metadata,
    SearchHit,
    TOCEntry,
)


def test_metadata_creation() -> None:
    m = Metadata(id="abc", name="test.pdf")
    assert m.id == "abc"
    assert m.name == "test.pdf"


def test_toc_entry() -> None:
    entry = TOCEntry(title="Intro", level=1, start_line=1, end_line=10)
    assert entry.level == 1
    assert entry.end_line == 10


def test_toc_entry_no_end() -> None:
    entry = TOCEntry(title="Intro", level=1, start_line=1)
    assert entry.end_line is None


def test_document_format_toc_empty() -> None:
    doc = Document(
        id="1", name="t.md", path_to_raw="/a", path_to_md="/b", num_lines=10
    )
    assert doc.format_toc() == "No table of contents available"


def test_document_format_toc() -> None:
    doc = Document(
        id="1",
        name="t.md",
        path_to_raw="/a",
        path_to_md="/b",
        num_lines=50,
        table_of_contents=[
            TOCEntry(title="Top", level=1, start_line=1, end_line=20),
            TOCEntry(title="Sub", level=2, start_line=5, end_line=20),
        ],
    )
    toc = doc.format_toc()
    assert "Top" in toc
    assert "  - Sub" in toc


def test_search_hit() -> None:
    hit = SearchHit(id="1", name="t.md", num_lines=10, table_of_contents="", score=0.9)
    assert hit.score == 0.9


def test_excerpt() -> None:
    e = Excerpt(doc_id="1", line_range=(1, 10), text="hello", total_lines=100)
    assert e.line_range == (1, 10)


def test_content_search_hit() -> None:
    h = ContentSearchHit(doc_id="1", line_range=(5, 15), text="match", score=0.8)
    assert h.score == 0.8


def test_chunk_metadata() -> None:
    c = ChunkMetadata(doc_id="1", chunk_index=0, start_line=1, end_line=80, text="content")
    assert c.chunk_index == 0
