"""Abstract base class for OCR providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class OCRProvider(ABC):
    """Base class for OCR/document-to-markdown converters."""

    @abstractmethod
    def convert(self, file_path: Path) -> str:
        """Convert a file to markdown text.

        Args:
            file_path: Path to the source file.

        Returns:
            Markdown string of the file contents.
        """

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return the set of file extensions this provider supports.

        Extensions should include the leading dot, e.g. ``{".pdf", ".docx"}``.
        """
