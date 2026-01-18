"""Tests for markdown output formatting."""

import tempfile
from pathlib import Path

from cv_warlock.output.markdown import format_match_analysis, format_result, save_markdown


class TestSaveMarkdown:
    """Tests for save_markdown function."""

    def test_saves_content_to_file(self) -> None:
        """Test that content is saved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.md"
            content = "# Test\n\nThis is a test."

            result = save_markdown(content, output_path)

            assert result == output_path
            assert output_path.exists()
            assert output_path.read_text(encoding="utf-8") == content

    def test_creates_parent_directories(self) -> None:
        """Test that parent directories are created if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "path" / "test.md"
            content = "# Test"

            result = save_markdown(content, output_path)

            assert result == output_path
            assert output_path.exists()

    def test_accepts_string_path(self) -> None:
        """Test that string paths are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test.md"
            content = "# Test"

            result = save_markdown(content, output_path)

            assert result == Path(output_path)
            assert Path(output_path).exists()

    def test_overwrites_existing_file(self) -> None:
        """Test that existing files are overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.md"

            # Write initial content
            save_markdown("Initial content", output_path)
            # Overwrite with new content
            save_markdown("New content", output_path)

            assert output_path.read_text(encoding="utf-8") == "New content"


class TestFormatMatchAnalysis:
    """Tests for format_match_analysis function."""

    def test_basic_match_analysis(self) -> None:
        """Test formatting of basic match analysis."""
        match_analysis = {
            "strong_matches": ["Python", "AWS"],
            "partial_matches": ["Docker"],
            "gaps": ["Kubernetes"],
            "transferable_skills": ["Leadership"],
            "relevance_score": 0.75,
        }

        result = format_match_analysis(match_analysis)

        assert "## Match Analysis (Score: 75%)" in result
        assert "### Strong Matches" in result
        assert "- Python" in result
        assert "- AWS" in result
        assert "### Partial Matches" in result
        assert "- Docker" in result
        assert "### Gaps" in result
        assert "- Kubernetes" in result
        assert "### Transferable Skills" in result
        assert "- Leadership" in result

    def test_empty_lists(self) -> None:
        """Test formatting with empty lists."""
        match_analysis = {
            "strong_matches": [],
            "partial_matches": [],
            "gaps": [],
            "transferable_skills": [],
            "relevance_score": 0.5,
        }

        result = format_match_analysis(match_analysis)

        assert "## Match Analysis (Score: 50%)" in result
        assert "### Strong Matches" in result
        assert "### Gaps" in result

    def test_hybrid_result_with_breakdown(self) -> None:
        """Test formatting of hybrid result with score breakdown."""
        match_analysis = {
            "strong_matches": ["Python"],
            "partial_matches": [],
            "gaps": ["Go"],
            "transferable_skills": [],
            "relevance_score": 0.80,
            "score_breakdown": {
                "exact_skill_match": 0.70,
                "semantic_skill_match": 0.85,
                "document_similarity": 0.75,
                "experience_years_fit": 0.90,
                "education_match": 0.80,
                "recency_score": 0.85,
            },
            "algorithmic_score": 0.78,
            "llm_adjustment": 0.02,
        }

        result = format_match_analysis(match_analysis)

        assert "### Score Breakdown" in result
        assert "**Exact Skill Match:** 70%" in result
        assert "**Semantic Skill Match:** 85%" in result
        assert "**Document Similarity:** 75%" in result
        assert "**Experience Years Fit:** 90%" in result
        assert "**Education Match:** 80%" in result
        assert "**Recency Score:** 85%" in result
        assert "Algorithmic: 78%, LLM adjustment: +2%" in result

    def test_knockout_triggered(self) -> None:
        """Test formatting when knockout is triggered."""
        match_analysis = {
            "strong_matches": [],
            "partial_matches": [],
            "gaps": ["Required skill missing"],
            "transferable_skills": [],
            "relevance_score": 0.30,
            "score_breakdown": {
                "exact_skill_match": 0.20,
                "semantic_skill_match": 0.30,
                "document_similarity": 0.40,
                "experience_years_fit": 0.50,
                "education_match": 0.60,
                "recency_score": 0.70,
            },
            "knockout_triggered": True,
            "knockout_reason": "Missing 3+ required skills",
        }

        result = format_match_analysis(match_analysis)

        assert "Knockout Triggered" in result
        assert "Missing 3+ required skills" in result

    def test_negative_llm_adjustment(self) -> None:
        """Test formatting with negative LLM adjustment."""
        match_analysis = {
            "strong_matches": [],
            "partial_matches": [],
            "gaps": [],
            "transferable_skills": [],
            "relevance_score": 0.70,
            "score_breakdown": {
                "exact_skill_match": 0.80,
                "semantic_skill_match": 0.80,
                "document_similarity": 0.80,
                "experience_years_fit": 0.80,
                "education_match": 0.80,
                "recency_score": 0.80,
            },
            "algorithmic_score": 0.80,
            "llm_adjustment": -0.10,
        }

        result = format_match_analysis(match_analysis)

        assert "LLM adjustment: -10%" in result


class TestFormatResult:
    """Tests for format_result function."""

    def test_with_errors(self) -> None:
        """Test formatting when errors exist."""
        state = {"errors": ["Error 1", "Error 2"]}

        result = format_result(state)

        assert "Errors occurred:" in result
        assert "- Error 1" in result
        assert "- Error 2" in result

    def test_with_match_analysis_and_cv(self) -> None:
        """Test formatting with match analysis and tailored CV."""
        state = {
            "errors": [],
            "match_analysis": {
                "strong_matches": ["Python"],
                "partial_matches": [],
                "gaps": [],
                "transferable_skills": [],
                "relevance_score": 0.80,
            },
            "tailored_cv": "# John Doe\n\nExperienced Engineer",
        }

        result = format_result(state)

        assert "## Match Analysis" in result
        assert "## Tailored CV" in result
        assert "# John Doe" in result

    def test_without_match_analysis(self) -> None:
        """Test formatting without match analysis."""
        state = {
            "errors": [],
            "tailored_cv": "# John Doe",
        }

        result = format_result(state)

        assert "## Tailored CV" in result
        assert "# John Doe" in result
        assert "Match Analysis" not in result

    def test_empty_state(self) -> None:
        """Test formatting with empty state."""
        state = {"errors": []}

        result = format_result(state)

        assert result == ""
