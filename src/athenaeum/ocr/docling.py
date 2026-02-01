"""OCR provider using Docling."""

from __future__ import annotations

from pathlib import Path

from athenaeum.ocr.base import OCRProvider


class DoclingOCR(OCRProvider):
    """Convert documents to markdown using Docling."""

    _EXTENSIONS = {".pdf", ".pptx", ".docx", ".xlsx", ".html", ".md"}

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as e:
            raise ImportError(
                "Docling is not installed. Install with: pip install 'athenaeum-kb[docling]'"
            ) from e
        self._converter = DocumentConverter()

    def convert(self, file_path: Path) -> str:
        result = self._converter.convert(str(file_path))
        md: str = result.document.export_to_markdown()
        return md

    def supported_extensions(self) -> set[str]:
        return self._EXTENSIONS
