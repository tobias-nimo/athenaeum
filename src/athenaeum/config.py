"""Configuration for Athenaeum."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class AthenaeumConfig:
    """Configuration for an Athenaeum instance."""

    storage_dir: Path = field(default_factory=lambda: Path.home() / ".athenaeum")
    chunk_size: int = 1500      # Characters per chunk
    chunk_overlap: int = 200    # Overlap in characters between consecutive chunks
    rrf_k: int = 60
    default_strategy: Literal["hybrid", "bm25", "vector"] = "hybrid"
    similarity_threshold: float | None = None  # Min cosine score [0,1]; None = no filter
