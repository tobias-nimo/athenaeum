"""Pydantic data models for Athenaeum."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    """Basic document metadata."""

    id: str = Field(..., description="Unique document identifier")
    name: str = Field(..., description="Document display name")


class TOCEntry(BaseModel):
    """Table of contents entry with line range."""

    title: str = Field(..., description="Section title")
    level: int = Field(..., description="Header level (1 for h1, 2 for h2)")
    start_line: int = Field(..., description="Starting line number (1-indexed)")
    end_line: int | None = Field(None, description="Ending line number (1-indexed, inclusive)")


class Document(BaseModel):
    """Full document record stored in the knowledge base."""

    id: str = Field(..., description="Unique document identifier (UUID)")
    name: str = Field(..., description="Original filename")
    path_to_raw: str = Field(..., description="Path to original file")
    path_to_md: str = Field(..., description="Path to converted markdown")
    num_lines: int = Field(..., description="Total number of lines in markdown")
    table_of_contents: list[TOCEntry] = Field(
        default_factory=list, description="Parsed table of contents"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    file_size: int = Field(0, description="Original file size in bytes")
    file_type: str = Field("", description="Original file extension")

    def format_toc(self) -> str:
        """Format table of contents as a readable string."""
        if not self.table_of_contents:
            return "No table of contents available"

        lines = []
        for entry in self.table_of_contents:
            indent = "  " * (entry.level - 1)
            line_info = f"[lines {entry.start_line}-{entry.end_line or '?'}]"
            lines.append(f"{indent}- {entry.title} {line_info}")
        return "\n".join(lines)


class SearchHit(BaseModel):
    """Search result for document-level search."""

    id: str = Field(..., description="Document identifier")
    name: str = Field(..., description="Document name")
    num_lines: int = Field(..., description="Total lines in document")
    table_of_contents: str = Field(..., description="Formatted table of contents")
    score: float = Field(default=0.0, description="Search relevance score")
    snippet: str = Field(default="", description="Relevant text snippet")


class Excerpt(BaseModel):
    """Text excerpt from a document."""

    doc_id: str = Field(..., description="Document identifier")
    line_range: tuple[int, int] = Field(..., description="Line range (start, end), 1-indexed")
    text: str = Field(..., description="Extracted text content")
    total_lines: int = Field(0, description="Total lines in document")


class ContentSearchHit(BaseModel):
    """Search result for within-document content search."""

    doc_id: str = Field(..., description="Document identifier")
    line_range: tuple[int, int] = Field(..., description="Line range of the match")
    text: str = Field(..., description="Matching text content")
    score: float = Field(0.0, description="Search relevance score")


class ChunkMetadata(BaseModel):
    """Metadata stored with each chunk in the vector store."""

    doc_id: str = Field(..., description="Parent document ID")
    chunk_index: int = Field(..., description="Chunk index within document")
    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")
    text: str = Field(..., description="Chunk text content")
