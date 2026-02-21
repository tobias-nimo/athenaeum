"""End-to-end tests for the Athenaeum orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from athenaeum import Athenaeum, AthenaeumConfig
from athenaeum.models import ContentSearchHit, SearchHit


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
    config = AthenaeumConfig(storage_dir=tmp_path / "athenaeum", chunk_size=200, chunk_overlap=50)
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


def test_load_doc_with_tags(athenaeum: Athenaeum, sample_md_path: Path) -> None:
    athenaeum.load_doc(str(sample_md_path), tags={"report", "finance"})
    docs = athenaeum.list_docs()
    assert len(docs) == 1
    assert docs[0].tags == {"report", "finance"}


def test_tag_doc(athenaeum: Athenaeum, sample_md_path: Path) -> None:
    doc_id = athenaeum.load_doc(str(sample_md_path))
    athenaeum.tag_doc(doc_id, {"new-tag", "another"})
    docs = athenaeum.list_docs()
    assert docs[0].tags == {"new-tag", "another"}


def test_untag_doc(athenaeum: Athenaeum, sample_md_path: Path) -> None:
    doc_id = athenaeum.load_doc(str(sample_md_path), tags={"a", "b", "c"})
    athenaeum.untag_doc(doc_id, {"b"})
    docs = athenaeum.list_docs()
    assert docs[0].tags == {"a", "c"}


def test_list_tags(
    athenaeum: Athenaeum, sample_md_path: Path, sample_txt_path: Path
) -> None:
    athenaeum.load_doc(str(sample_md_path), tags={"alpha", "beta"})
    athenaeum.load_doc(str(sample_txt_path), tags={"beta", "gamma"})
    assert athenaeum.list_tags() == {"alpha", "beta", "gamma"}


def test_list_docs_filter_by_tags(
    athenaeum: Athenaeum, sample_md_path: Path, sample_txt_path: Path
) -> None:
    athenaeum.load_doc(str(sample_md_path), tags={"report"})
    athenaeum.load_doc(str(sample_txt_path), tags={"notes"})
    filtered = athenaeum.list_docs(tags={"report"})
    assert len(filtered) == 1
    assert filtered[0].name == "sample.md"


def test_search_docs_filter_by_tags(
    athenaeum: Athenaeum, sample_md_path: Path, sample_txt_path: Path
) -> None:
    athenaeum.load_doc(str(sample_md_path), tags={"report"})
    athenaeum.load_doc(str(sample_txt_path), tags={"notes"})
    results = athenaeum.search_docs("sample", scope="names", tags={"report"})
    assert len(results) == 1
    assert results[0].name == "sample.md"


def test_search_docs_multiple_tags_or(
    athenaeum: Athenaeum, sample_md_path: Path, sample_txt_path: Path
) -> None:
    athenaeum.load_doc(str(sample_md_path), tags={"report"})
    athenaeum.load_doc(str(sample_txt_path), tags={"notes"})
    results = athenaeum.search_docs("sample", scope="names", tags={"report", "notes"})
    assert len(results) == 2


def test_search_docs_no_tags_returns_all(
    athenaeum: Athenaeum, sample_md_path: Path, sample_txt_path: Path
) -> None:
    athenaeum.load_doc(str(sample_md_path), tags={"report"})
    athenaeum.load_doc(str(sample_txt_path), tags={"notes"})
    results = athenaeum.search_docs("sample", scope="names")
    assert len(results) == 2


# --- similarity_threshold tests ---


def test_similarity_threshold_filters_all(tmp_path: Path, sample_md_path: Path) -> None:
    config = AthenaeumConfig(
        storage_dir=tmp_path / "athenaeum",
        chunk_size=200,
        chunk_overlap=50,
        similarity_threshold=1.0,
    )
    kb = Athenaeum(embeddings=FakeEmbeddings(), config=config)
    kb.load_doc(str(sample_md_path))
    results = kb.search_docs("search strategies", strategy="vector")
    assert results == []


def test_similarity_threshold_zero_passes_all(tmp_path: Path, sample_md_path: Path) -> None:
    config_thresh = AthenaeumConfig(
        storage_dir=tmp_path / "thresh",
        chunk_size=200,
        chunk_overlap=50,
        similarity_threshold=0.0,
    )
    config_none = AthenaeumConfig(
        storage_dir=tmp_path / "none",
        chunk_size=200,
        chunk_overlap=50,
        similarity_threshold=None,
    )
    kb_thresh = Athenaeum(embeddings=FakeEmbeddings(), config=config_thresh)
    kb_none = Athenaeum(embeddings=FakeEmbeddings(), config=config_none)
    kb_thresh.load_doc(str(sample_md_path))
    kb_none.load_doc(str(sample_md_path))
    results_thresh = kb_thresh.search_docs("search", strategy="vector")
    results_none = kb_none.search_docs("search", strategy="vector")
    assert len(results_thresh) == len(results_none)


# --- aggregate=False tests ---


def test_search_docs_aggregate_false_returns_content_hits(
    athenaeum: Athenaeum, sample_md_path: Path
) -> None:
    athenaeum.load_doc(str(sample_md_path))
    results = athenaeum.search_docs("search strategies", strategy="bm25", aggregate=False)
    assert len(results) > 0
    assert all(isinstance(r, ContentSearchHit) for r in results)


def test_search_docs_aggregate_false_name_populated(
    athenaeum: Athenaeum, sample_md_path: Path
) -> None:
    athenaeum.load_doc(str(sample_md_path))
    results = athenaeum.search_docs("search strategies", strategy="bm25", aggregate=False)
    assert len(results) > 0
    assert all(r.name == "sample.md" for r in results)


def test_search_docs_aggregate_false_respects_top_k(
    athenaeum: Athenaeum, sample_md_path: Path
) -> None:
    athenaeum.load_doc(str(sample_md_path))
    top_k = 2
    results = athenaeum.search_docs("search", strategy="bm25", top_k=top_k, aggregate=False)
    assert len(results) <= top_k


def test_search_docs_aggregate_false_with_tags(
    athenaeum: Athenaeum, sample_md_path: Path, sample_txt_path: Path
) -> None:
    athenaeum.load_doc(str(sample_md_path), tags={"report"})
    athenaeum.load_doc(str(sample_txt_path), tags={"notes"})
    results = athenaeum.search_docs(
        "search", strategy="bm25", tags={"report"}, aggregate=False
    )
    assert all(r.name == "sample.md" for r in results)


def test_search_docs_aggregate_true_default(
    athenaeum: Athenaeum, sample_md_path: Path
) -> None:
    athenaeum.load_doc(str(sample_md_path))
    results = athenaeum.search_docs("search strategies", strategy="bm25")
    assert len(results) > 0
    assert all(isinstance(r, SearchHit) for r in results)
