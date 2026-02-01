"""Tests for StorageManager."""

from pathlib import Path

from athenaeum.storage import StorageManager


def test_storage_creates_root(tmp_path: Path) -> None:
    root = tmp_path / "storage"
    sm = StorageManager(root)
    assert root.exists()
    assert sm.root == root


def test_doc_dir(tmp_path: Path) -> None:
    sm = StorageManager(tmp_path / "storage")
    d = sm.doc_dir("abc")
    assert d.exists()
    assert d.name == "abc"


def test_raw_and_content_paths(tmp_path: Path) -> None:
    sm = StorageManager(tmp_path / "storage")
    raw = sm.raw_path("doc1", ".pdf")
    assert raw.name == "raw.pdf"
    md = sm.content_md_path("doc1")
    assert md.name == "content.md"


def test_remove_doc(tmp_path: Path) -> None:
    sm = StorageManager(tmp_path / "storage")
    sm.doc_dir("doc1")
    sm.raw_path("doc1", ".txt").write_text("test")
    sm.remove_doc("doc1")
    assert not (sm.docs_dir / "doc1").exists()


def test_remove_doc_nonexistent(tmp_path: Path) -> None:
    sm = StorageManager(tmp_path / "storage")
    sm.remove_doc("nonexistent")  # should not raise


def test_chroma_dir(tmp_path: Path) -> None:
    sm = StorageManager(tmp_path / "storage")
    cd = sm.ensure_chroma_dir()
    assert cd.exists()


def test_metadata_roundtrip(tmp_path: Path) -> None:
    sm = StorageManager(tmp_path / "storage")
    data = {"docs": {"doc1": {"name": "test.pdf"}}}
    sm.save_metadata(data)
    loaded = sm.load_metadata()
    assert loaded == data


def test_metadata_empty(tmp_path: Path) -> None:
    sm = StorageManager(tmp_path / "storage")
    assert sm.load_metadata() == {}
