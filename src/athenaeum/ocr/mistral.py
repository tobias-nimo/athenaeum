"""OCR provider using Mistral OCR API."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from athenaeum.ocr.base import OCRProvider


class MistralOCR(OCRProvider):
    """Convert documents to markdown using Mistral's OCR API."""

    _EXTENSIONS = {".pdf"}

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from mistralai import Mistral
        except ImportError as e:
            raise ImportError(
                "Mistral SDK is not installed. Install with: pip install 'athenaeum-kb[mistral]'"
            ) from e
        key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not key:
            raise ValueError("MISTRAL_API_KEY must be set or passed explicitly.")
        self._client = Mistral(api_key=key)

    def convert(self, file_path: Path) -> str:
        from mistralai import DocumentURLChunk, OCRRequest

        encoded = base64.standard_b64encode(file_path.read_bytes()).decode()
        data_uri = f"data:application/pdf;base64,{encoded}"
        response = self._client.ocr.process(
            request=OCRRequest(
                document=DocumentURLChunk(document_url=data_uri),
                model="mistral-ocr-latest",
            )
        )
        pages = [page.markdown for page in response.pages]
        return "\n\n".join(pages)

    def supported_extensions(self) -> set[str]:
        return self._EXTENSIONS
