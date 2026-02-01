"""Configuration for Athenaeum."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class AthenaeumConfig:
    """Configuration for an Athenaeum instance."""

    storage_dir: Path = field(default_factory=lambda: Path.home() / ".athenaeum")
    chunk_size: int = 80
    chunk_overlap: int = 20
    rrf_k: int = 60
    default_strategy: Literal["hybrid", "bm25", "vector"] = "hybrid"
