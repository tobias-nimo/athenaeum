"""Storage layout manager for Athenaeum."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


class StorageManager:
    """Manages the on-disk layout under the storage root.

    Layout::

        <root>/
            docs/<doc_id>/raw.*       # original file
            docs/<doc_id>/content.md  # converted markdown
            index/chroma/             # Chroma persistent directory
            metadata.json             # document registry
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def docs_dir(self) -> Path:
        return self.root / "docs"

    @property
    def chroma_dir(self) -> Path:
        return self.root / "index" / "chroma"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    def doc_dir(self, doc_id: str) -> Path:
        d = self.docs_dir / doc_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def raw_path(self, doc_id: str, suffix: str) -> Path:
        """Return path for storing the original file."""
        return self.doc_dir(doc_id) / f"raw{suffix}"

    def content_md_path(self, doc_id: str) -> Path:
        """Return path for the converted markdown."""
        return self.doc_dir(doc_id) / "content.md"

    def remove_doc(self, doc_id: str) -> None:
        """Remove a document's directory."""
        d = self.docs_dir / doc_id
        if d.exists():
            shutil.rmtree(d)

    def ensure_chroma_dir(self) -> Path:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        return self.chroma_dir

    def load_metadata(self) -> dict[str, Any]:
        if self.metadata_path.exists():
            return json.loads(self.metadata_path.read_text())  # type: ignore[no-any-return]
        return {}

    def save_metadata(self, data: dict[str, Any]) -> None:
        self.metadata_path.write_text(json.dumps(data, indent=2, default=str))
