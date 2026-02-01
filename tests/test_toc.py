"""Tests for TOC extraction."""

from athenaeum.toc import extract_toc


def test_extract_toc_basic(sample_md_text: str) -> None:
    entries = extract_toc(sample_md_text)
    titles = [e.title for e in entries]
    assert "Introduction" in titles
    assert "Getting Started" in titles
    assert "Features" in titles
    assert "Search" in titles
    assert "Document Ingestion" in titles
    assert "Conclusion" in titles


def test_extract_toc_levels(sample_md_text: str) -> None:
    entries = extract_toc(sample_md_text)
    by_title = {e.title: e for e in entries}
    assert by_title["Introduction"].level == 1
    assert by_title["Getting Started"].level == 2
    assert by_title["Search"].level == 3


def test_extract_toc_line_ranges(sample_md_text: str) -> None:
    entries = extract_toc(sample_md_text)
    for entry in entries:
        assert entry.start_line >= 1
        assert entry.end_line is not None
        assert entry.end_line >= entry.start_line


def test_extract_toc_h1_spans_to_end(sample_md_text: str) -> None:
    entries = extract_toc(sample_md_text)
    h1 = [e for e in entries if e.level == 1]
    assert len(h1) == 1
    total_lines = len(sample_md_text.split("\n"))
    assert h1[0].end_line == total_lines


def test_extract_toc_empty() -> None:
    assert extract_toc("") == []
    assert extract_toc("No headings here\nJust text") == []
