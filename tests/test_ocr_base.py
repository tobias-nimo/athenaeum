"""Tests for OCR providers."""

from pathlib import Path

import pytest

from athenaeum.ocr import get_ocr_provider
from athenaeum.ocr.base import OCRProvider
from athenaeum.ocr.custom import CustomOCR
from athenaeum.ocr.markitdown import MarkitdownOCR


def test_markitdown_supported_extensions() -> None:
    m = MarkitdownOCR()
    exts = m.supported_extensions()
    assert ".pdf" in exts
    assert ".md" in exts
    assert ".docx" in exts


def test_markitdown_convert_md(sample_md_path: Path) -> None:
    m = MarkitdownOCR()
    result = m.convert(sample_md_path)
    assert "Introduction" in result
    assert len(result) > 0


def test_markitdown_convert_txt(sample_txt_path: Path) -> None:
    m = MarkitdownOCR()
    result = m.convert(sample_txt_path)
    assert "plain text" in result


def test_custom_ocr() -> None:
    def fn(p):  # type: ignore[no-untyped-def]
        return f"converted: {p.name}"

    ocr = CustomOCR(fn, extensions={".txt"})
    assert ocr.supported_extensions() == {".txt"}
    result = ocr.convert(Path("test.txt"))
    assert result == "converted: test.txt"


def test_custom_ocr_default_extensions() -> None:
    ocr = CustomOCR(lambda p: "")
    assert ".*" in ocr.supported_extensions()


def test_factory_markitdown() -> None:
    provider = get_ocr_provider("markitdown")
    assert isinstance(provider, MarkitdownOCR)


def test_factory_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown OCR backend"):
        get_ocr_provider("nonexistent")  # type: ignore[arg-type]


def test_ocr_provider_is_abc() -> None:
    with pytest.raises(TypeError):
        OCRProvider()  # type: ignore[abstract]
