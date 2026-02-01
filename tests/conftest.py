"""Shared test fixtures for Athenaeum."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_md_path() -> Path:
    return FIXTURES_DIR / "sample.md"


@pytest.fixture
def sample_txt_path() -> Path:
    return FIXTURES_DIR / "sample.txt"


@pytest.fixture
def sample_md_text(sample_md_path: Path) -> str:
    return sample_md_path.read_text()
