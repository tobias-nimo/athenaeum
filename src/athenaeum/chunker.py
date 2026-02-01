"""Line-aware markdown chunking with heading-boundary snapping."""

from __future__ import annotations

import re

from athenaeum.models import ChunkMetadata

_HEADING_RE = re.compile(r"^#{1,6}\s+")


def _find_heading_lines(lines: list[str]) -> set[int]:
    """Return 0-indexed line numbers that are markdown headings."""
    return {i for i, line in enumerate(lines) if _HEADING_RE.match(line.strip())}


def chunk_markdown(
    markdown: str,
    doc_id: str,
    chunk_size: int = 80,
    chunk_overlap: int = 20,
) -> list[ChunkMetadata]:
    """Split markdown into overlapping, line-based chunks.

    Chunks are snapped to heading boundaries when a heading falls within the
    overlap region, so sections start cleanly.

    Args:
        markdown: Full markdown text.
        doc_id: Parent document ID.
        chunk_size: Target number of lines per chunk.
        chunk_overlap: Number of overlapping lines between consecutive chunks.

    Returns:
        List of ``ChunkMetadata`` instances.
    """
    if not markdown:
        return []

    lines = markdown.split("\n")
    total = len(lines)

    heading_lines = _find_heading_lines(lines)
    chunks: list[ChunkMetadata] = []
    start = 0
    chunk_index = 0

    while start < total:
        end = min(start + chunk_size, total)
        text = "\n".join(lines[start:end])

        chunks.append(
            ChunkMetadata(
                doc_id=doc_id,
                chunk_index=chunk_index,
                start_line=start + 1,  # 1-indexed
                end_line=end,  # 1-indexed inclusive
                text=text,
            )
        )
        chunk_index += 1

        if end >= total:
            break

        # Compute next start with overlap
        next_start = end - chunk_overlap

        # Snap to heading if one exists in the overlap zone [next_start, end)
        for line_idx in range(next_start, end):
            if line_idx in heading_lines:
                next_start = line_idx
                break

        start = next_start

    return chunks
