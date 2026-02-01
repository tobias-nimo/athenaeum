"""End-to-end tests for the Athenaeum orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from athenaeum import Athenaeum, AthenaeumConfig


class FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        vec = [0.0] * 32
        for i, ch in enumerate(text.encode()):
            vec[i % 32] += ch / 256.0
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


@pytest.fixture
def athenaeum(tmp_path: Path) -> Athenaeum:
    config = AthenaeumConfig(storage_dir=tmp_path / "athenaeum", chunk_size=10, chunk_overlap=2)
    return Athenaeum(embeddings=FakeEmbeddings(), config=config)


def test_load_doc_md(athenaeum: Athenaeum, sample_md_path: Path) -> None:
    doc_id = athenaeum.load_doc(str(sample_md_path))
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0


def test_load_doc_txt(athenaeum: Athenaeum, sample_txt_path: Path) -> None:
    doc_id = athenaeum.load_doc(str(sample_txt_path))
    assert isinstance(doc_id, str)


def test_load_doc_not_found(athenaeum: Athenaeum) -> None:
    with pytest.raises(FileNotFoundError):
        athenaeum.load_doc("/nonexistent/file.md")


def test_list_docs(athenaeum: Athenaeum, sample_md_path: Path) -> None:
    athenaeum.load_doc(str(sample_md_path))
    docs = athenaeum.list_docs()
    assert len(docs) == 1
    assert docs[0].name == "sample.md"


def test_search_docs_contents(athenaeum: Athenaeum, sample_md_path: Path) -> None:
    athenaeum.load_doc(str(sample_md_path))
    results = athenaeum.search_docs("search strategies", strategy="bm25")
    assert len(results) > 0


def test_search_docs_names(athenaeum: Athenaeum, sample_md_path: Path) -> None:
    athenaeum.load_doc(str(sample_md_path))
    results = athenaeum.search_docs("sample", scope="names")
    assert len(results) == 1
    assert results[0].name == "sample.md"


def test_search_docs_names_no_match(athenaeum: Athenaeum, sample_md_path: Path) -> None:
    athenaeum.load_doc(str(sample_md_path))
    results = athenaeum.search_docs("nonexistent", scope="names")
    assert len(results) == 0


def test_search_doc_contents(athenaeum: Athenaeum, sample_md_path: Path) -> None:
    doc_id = athenaeum.load_doc(str(sample_md_path))
    results = athenaeum.search_doc_contents(doc_id, "BM25 keyword search", strategy="bm25")
    assert len(results) > 0
    assert results[0].doc_id == doc_id


def test_search_doc_contents_not_found(athenaeum: Athenaeum) -> None:
    with pytest.raises(ValueError, match="Document not found"):
        athenaeum.search_doc_contents("fake-id", "query")


def test_read_doc(athenaeum: Athenaeum, sample_md_path: Path) -> None:
    doc_id = athenaeum.load_doc(str(sample_md_path))
    excerpt = athenaeum.read_doc(doc_id, start_line=1, end_line=5)
    assert excerpt.doc_id == doc_id
    assert excerpt.line_range == (1, 5)
    assert "Introduction" in excerpt.text
    assert excerpt.total_lines > 0


def test_read_doc_not_found(athenaeum: Athenaeum) -> None:
    with pytest.raises(ValueError, match="Document not found"):
        athenaeum.read_doc("fake-id")


def test_search_docs_vector(athenaeum: Athenaeum, sample_md_path: Path) -> None:
    athenaeum.load_doc(str(sample_md_path))
    results = athenaeum.search_docs("search", strategy="vector")
    assert len(results) > 0


def test_search_docs_hybrid(athenaeum: Athenaeum, sample_md_path: Path) -> None:
    athenaeum.load_doc(str(sample_md_path))
    results = athenaeum.search_docs("search", strategy="hybrid")
    assert len(results) > 0


def test_multiple_docs(
    athenaeum: Athenaeum, sample_md_path: Path, sample_txt_path: Path
) -> None:
    athenaeum.load_doc(str(sample_md_path))
    athenaeum.load_doc(str(sample_txt_path))
    docs = athenaeum.list_docs()
    assert len(docs) == 2
