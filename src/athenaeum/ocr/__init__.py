"""OCR provider registry and factory."""

from __future__ import annotations

from typing import Literal

from athenaeum.ocr.base import OCRProvider
from athenaeum.ocr.custom import CustomOCR
from athenaeum.ocr.markitdown import MarkitdownOCR

OCRBackend = Literal["markitdown", "docling", "mistral", "lighton"]


def get_ocr_provider(backend: OCRBackend = "markitdown", **kwargs: object) -> OCRProvider:
    """Factory to create an OCR provider by name.

    Args:
        backend: One of ``"markitdown"``, ``"docling"``, ``"mistral"``, ``"lighton"``.
        **kwargs: Passed to the provider constructor.

    Returns:
        An ``OCRProvider`` instance.
    """
    if backend == "markitdown":
        return MarkitdownOCR()
    if backend == "docling":
        from athenaeum.ocr.docling import DoclingOCR

        return DoclingOCR()
    if backend == "mistral":
        from athenaeum.ocr.mistral import MistralOCR

        return MistralOCR(**kwargs)  # type: ignore[arg-type]
    if backend == "lighton":
        from athenaeum.ocr.lighton import LightOnOCR

        return LightOnOCR(**kwargs)  # type: ignore[arg-type]

    raise ValueError(f"Unknown OCR backend: {backend!r}")


__all__ = [
    "CustomOCR",
    "MarkitdownOCR",
    "OCRBackend",
    "OCRProvider",
    "get_ocr_provider",
]
