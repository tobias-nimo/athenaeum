"""Table of contents extraction from markdown headings."""

from __future__ import annotations

import re

from athenaeum.models import TOCEntry

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def extract_toc(markdown: str) -> list[TOCEntry]:
    """Extract a table of contents from markdown heading lines.

    Each entry records the heading level, title, and the line range it spans.
    Line numbers are 1-indexed. The ``end_line`` of each entry is set to the
    line before the next heading at the same or higher level, or to the last
    line of the document for the final entry.
    """
    lines = markdown.split("\n")
    entries: list[TOCEntry] = []

    for line_no_0, line in enumerate(lines):
        m = _HEADING_RE.match(line.strip())
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            entries.append(
                TOCEntry(title=title, level=level, start_line=line_no_0 + 1, end_line=None)
            )

    # Fill in end_line for each entry
    total_lines = len(lines)
    for i, entry in enumerate(entries):
        # Find next entry at same or higher (lower number) level
        end = total_lines
        for j in range(i + 1, len(entries)):
            if entries[j].level <= entry.level:
                end = entries[j].start_line - 1
                break
        else:
            # Last entry at this level — ends at document end
            if i + 1 < len(entries):
                end = entries[-1].start_line - 1
                # Actually, just go to end of document
                end = total_lines
        entry.end_line = end

    return entries
