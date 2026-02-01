"""OCR provider using Microsoft markitdown."""

from __future__ import annotations

from pathlib import Path

from markitdown import MarkItDown

from athenaeum.ocr.base import OCRProvider


class MarkitdownOCR(OCRProvider):
    """Convert documents to markdown using markitdown."""

    _EXTENSIONS = {
        ".pdf", ".pptx", ".docx", ".xlsx",
        ".json", ".csv", ".txt", ".md",
        ".html", ".xml", ".rtf", ".epub",
    }

    def __init__(self) -> None:
        self._converter = MarkItDown()

    def convert(self, file_path: Path) -> str:
        result = self._converter.convert(str(file_path))
        return result.text_content

    def supported_extensions(self) -> set[str]:
        return self._EXTENSIONS
