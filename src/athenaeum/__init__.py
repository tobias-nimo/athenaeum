"""Athenaeum - Tools for intelligent interaction with knowledge bases."""

from athenaeum.athenaeum import Athenaeum
from athenaeum.config import AthenaeumConfig
from athenaeum.models import (
    ChunkMetadata,
    ContentSearchHit,
    Document,
    Excerpt,
    Metadata,
    SearchHit,
    TOCEntry,
)
from athenaeum.ocr import OCRProvider, get_ocr_provider

__all__ = [
    "Athenaeum",
    "AthenaeumConfig",
    "ChunkMetadata",
    "ContentSearchHit",
    "Document",
    "Excerpt",
    "Metadata",
    "OCRProvider",
    "SearchHit",
    "TOCEntry",
    "get_ocr_provider",
]
