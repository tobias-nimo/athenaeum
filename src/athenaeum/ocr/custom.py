"""Custom OCR provider wrapping a user-supplied callable."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from athenaeum.ocr.base import OCRProvider


class CustomOCR(OCRProvider):
    """Wrap an arbitrary ``(Path) -> str`` callable as an OCR provider."""

    def __init__(
        self,
        fn: Callable[[Path], str],
        extensions: set[str] | None = None,
    ) -> None:
        self._fn = fn
        self._extensions = extensions or {".*"}

    def convert(self, file_path: Path) -> str:
        return self._fn(file_path)

    def supported_extensions(self) -> set[str]:
        return self._extensions
