"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def sample_cv_text() -> str:
    """Load sample CV text."""
    sample_path = Path(__file__).parent.parent / "examples" / "sample_cv.md"
    return sample_path.read_text(encoding="utf-8")


@pytest.fixture
def sample_job_text() -> str:
    """Load sample job posting text."""
    sample_path = Path(__file__).parent.parent / "examples" / "sample_job_posting.md"
    return sample_path.read_text(encoding="utf-8")


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"
