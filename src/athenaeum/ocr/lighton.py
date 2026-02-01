"""OCR provider using LightOnOCR-2-1B."""

from __future__ import annotations

from pathlib import Path

from athenaeum.ocr.base import OCRProvider


class LightOnOCR(OCRProvider):
    """Convert documents to markdown using LightOnOCR-2-1B (local model)."""

    _EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

    def __init__(self, device: str = "cpu") -> None:
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "transformers/torch not installed. "
                "Install with: pip install 'athenaeum-kb[lighton]'"
            ) from e
        self._pipe = pipeline(
            "image-text-to-text",
            model="lightonai/LightOnOCR-2-1B",
            device=device,
        )

    def convert(self, file_path: Path) -> str:
        from PIL import Image

        image = Image.open(file_path)
        result = self._pipe(image)
        if isinstance(result, list) and len(result) > 0:
            text: str = result[0].get("generated_text", "")
            return text
        return str(result)

    def supported_extensions(self) -> set[str]:
        return self._EXTENSIONS
