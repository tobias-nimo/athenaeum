"""JSON-backed document registry."""

from __future__ import annotations

from athenaeum.models import Document
from athenaeum.storage import StorageManager


class DocumentStore:
    """Manages the document registry backed by ``metadata.json``."""

    def __init__(self, storage: StorageManager) -> None:
        self._storage = storage
        self._docs: dict[str, Document] = {}
        self._load()

    def _load(self) -> None:
        raw = self._storage.load_metadata()
        docs_raw = raw.get("documents", {})
        for doc_id, data in docs_raw.items():
            self._docs[doc_id] = Document.model_validate(data)

    def _save(self) -> None:
        data = {
            "documents": {
                doc_id: doc.model_dump(mode="json") for doc_id, doc in self._docs.items()
            }
        }
        self._storage.save_metadata(data)

    def add(self, doc: Document) -> None:
        """Add or update a document in the registry."""
        self._docs[doc.id] = doc
        self._save()

    def get(self, doc_id: str) -> Document | None:
        """Get a document by ID, or None if not found."""
        return self._docs.get(doc_id)

    def list_all(self) -> list[Document]:
        """Return all documents."""
        return list(self._docs.values())

    def remove(self, doc_id: str) -> Document | None:
        """Remove a document from the registry. Returns the removed doc or None."""
        doc = self._docs.pop(doc_id, None)
        if doc is not None:
            self._save()
        return doc

    def list_by_tags(self, tags: set[str]) -> list[Document]:
        """Return documents matching ANY of the given tags (OR semantics)."""
        return [doc for doc in self._docs.values() if doc.tags & tags]

    def list_tags(self) -> set[str]:
        """Return the union of all tags across all documents."""
        result: set[str] = set()
        for doc in self._docs.values():
            result |= doc.tags
        return result

    @property
    def count(self) -> int:
        return len(self._docs)
