"""Tests for PDF generator."""

import pytest

from app.utils.pdf_generator import CVPDFGenerator, _sanitize_markdown_bold


class TestCVPDFGenerator:
    """Tests for CVPDFGenerator class."""

    def test_safe_multi_cell_handles_narrow_width(self) -> None:
        """Test that _safe_multi_cell handles narrow width by forcing full width."""
        pdf = CVPDFGenerator()
        pdf.add_page()

        # Extremely narrow width (10 units)
        # Content "WideWord" (likely wider than 10 units)
        long_word = "Supercalifragilisticexpialidocious"
        pdf.set_font("Helvetica", size=12)

        # Should not raise error
        try:
            pdf._safe_multi_cell(10, 5, long_word)
        except Exception as e:
            pytest.fail(f"_safe_multi_cell failed with narrow width: {e}")

        # Verify it moved to a new line (y increased) or at least content was written
        # Accessing PDF content is hard, but absence of crash is the main test here

    def test_safe_multi_cell_handles_wide_word(self) -> None:
        """Test that _safe_multi_cell handles word wider than column."""
        pdf = CVPDFGenerator()
        pdf.add_page()

        # Moderate width, but very long word
        width = 30
        long_word = "A" * 50  # Very long word
        pdf.set_font("Helvetica", size=12)

        try:
            pdf._safe_multi_cell(width, 5, long_word)
        except Exception as e:
            pytest.fail(f"_safe_multi_cell failed with wide word: {e}")

    def test_sanitize_markdown_bold(self) -> None:
        """Test markdown bold sanitization."""
        # Fix *text:** -> **text:**
        assert _sanitize_markdown_bold("*Skills:**") == "**Skills:**"
        # Fix **text:* -> **text:**
        assert _sanitize_markdown_bold("**Skills:*") == "**Skills:**"
        # Fix *Category:** -> **Category:**
        assert _sanitize_markdown_bold("*Languages:**") == "**Languages:**"
        # Leave correct ones alone
        assert _sanitize_markdown_bold("**Correct:**") == "**Correct:**"
        # Leave normal text alone
        assert _sanitize_markdown_bold("Normal text") == "Normal text"
