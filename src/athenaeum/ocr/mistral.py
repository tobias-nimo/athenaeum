"""OCR provider using Mistral OCR API."""

from __future__ import annotations

import os
from pathlib import Path

from athenaeum.ocr.base import OCRProvider


class MistralOCR(OCRProvider):
    """Convert documents to markdown using Mistral's OCR API."""

    _EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".avif"}

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
        # Upload file to Mistral Cloud
        uploaded_file = self._client.files.upload(
            file={
                "file_name": file_path.name,
                "content": file_path.read_bytes(),
            },
            purpose="ocr",
        )

        try:
            # Get a signed URL to access the file
            signed_url = self._client.files.get_signed_url(file_id=uploaded_file.id)

            # Determine document type based on file extension
            ext = file_path.suffix.lower()
            if ext in {".png", ".jpg", ".jpeg", ".avif"}:
                doc_type = "image_url"
                url_key = "image_url"
            else:
                doc_type = "document_url"
                url_key = "document_url"

            # Process with OCR
            response = self._client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": doc_type,
                    url_key: signed_url.url,
                },
            )

            pages = [page.markdown for page in response.pages]
            return "\n\n".join(pages)
        finally:
            # Clean up: delete the file from Mistral Cloud
            self._client.files.delete(file_id=uploaded_file.id)

    def supported_extensions(self) -> set[str]:
        return self._EXTENSIONS
