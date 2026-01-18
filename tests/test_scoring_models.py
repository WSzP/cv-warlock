"""Tests for the scoring models."""

import pytest
from pydantic import ValidationError

from cv_warlock.scoring.models import (
    AlgorithmicScores,
    HybridMatchResult,
    LLMAssessmentOutput,
    ScoreBreakdown,
)


class TestScoreBreakdown:
    """Tests for ScoreBreakdown TypedDict."""

    def test_create_valid_breakdown(self) -> None:
        """Test creating a valid ScoreBreakdown."""
        breakdown: ScoreBreakdown = {
            "exact_skill_match": 0.85,
            "semantic_skill_match": 0.75,
            "document_similarity": 0.80,
            "experience_years_fit": 0.90,
            "education_match": 1.0,
            "recency_score": 0.70,
        }

        assert breakdown["exact_skill_match"] == 0.85
        assert breakdown["semantic_skill_match"] == 0.75
        assert breakdown["document_similarity"] == 0.80
        assert breakdown["experience_years_fit"] == 0.90
        assert breakdown["education_match"] == 1.0
        assert breakdown["recency_score"] == 0.70

    def test_breakdown_with_zero_scores(self) -> None:
        """Test ScoreBreakdown with all zero scores."""
        breakdown: ScoreBreakdown = {
            "exact_skill_match": 0.0,
            "semantic_skill_match": 0.0,
            "document_similarity": 0.0,
            "experience_years_fit": 0.0,
            "education_match": 0.0,
            "recency_score": 0.0,
        }

        assert all(v == 0.0 for v in breakdown.values())

    def test_breakdown_with_max_scores(self) -> None:
        """Test ScoreBreakdown with all max scores."""
        breakdown: ScoreBreakdown = {
            "exact_skill_match": 1.0,
            "semantic_skill_match": 1.0,
            "document_similarity": 1.0,
            "experience_years_fit": 1.0,
            "education_match": 1.0,
            "recency_score": 1.0,
        }

        assert all(v == 1.0 for v in breakdown.values())


class TestAlgorithmicScores:
    """Tests for AlgorithmicScores Pydantic model."""

    def test_create_valid_scores(self) -> None:
        """Test creating valid AlgorithmicScores."""
        scores = AlgorithmicScores(
            exact_skill_match=0.85,
            semantic_skill_match=0.75,
            document_similarity=0.80,
            experience_years_fit=0.90,
            education_match=1.0,
            recency_score=0.70,
            total=0.82,
        )

        assert scores.exact_skill_match == 0.85
        assert scores.total == 0.82
        assert scores.knockout_triggered is False
        assert scores.knockout_reason is None

    def test_create_scores_with_knockout(self) -> None:
        """Test AlgorithmicScores with knockout triggered."""
        scores = AlgorithmicScores(
            exact_skill_match=0.0,
            semantic_skill_match=0.0,
            document_similarity=0.3,
            experience_years_fit=0.5,
            education_match=1.0,
            recency_score=0.4,
            total=0.0,
            knockout_triggered=True,
            knockout_reason="Missing required skills: Python, AWS",
        )

        assert scores.knockout_triggered is True
        assert "Python" in scores.knockout_reason
        assert scores.total == 0.0

    def test_scores_boundary_validation_zero(self) -> None:
        """Test that zero values are valid."""
        scores = AlgorithmicScores(
            exact_skill_match=0.0,
            semantic_skill_match=0.0,
            document_similarity=0.0,
            experience_years_fit=0.0,
            education_match=0.0,
            recency_score=0.0,
            total=0.0,
        )

        assert scores.exact_skill_match == 0.0

    def test_scores_boundary_validation_one(self) -> None:
        """Test that 1.0 values are valid."""
        scores = AlgorithmicScores(
            exact_skill_match=1.0,
            semantic_skill_match=1.0,
            document_similarity=1.0,
            experience_years_fit=1.0,
            education_match=1.0,
            recency_score=1.0,
            total=1.0,
        )

        assert scores.total == 1.0

    def test_scores_validation_negative_value(self) -> None:
        """Test that negative values raise validation error."""
        with pytest.raises(ValidationError):
            AlgorithmicScores(
                exact_skill_match=-0.1,
                semantic_skill_match=0.75,
                document_similarity=0.80,
                experience_years_fit=0.90,
                education_match=1.0,
                recency_score=0.70,
                total=0.5,
            )

    def test_scores_validation_exceeds_one(self) -> None:
        """Test that values > 1 raise validation error."""
        with pytest.raises(ValidationError):
            AlgorithmicScores(
                exact_skill_match=1.1,
                semantic_skill_match=0.75,
                document_similarity=0.80,
                experience_years_fit=0.90,
                education_match=1.0,
                recency_score=0.70,
                total=0.5,
            )

    def test_to_breakdown_method(self) -> None:
        """Test converting AlgorithmicScores to ScoreBreakdown."""
        scores = AlgorithmicScores(
            exact_skill_match=0.85,
            semantic_skill_match=0.75,
            document_similarity=0.80,
            experience_years_fit=0.90,
            education_match=1.0,
            recency_score=0.70,
            total=0.82,
        )

        breakdown = scores.to_breakdown()

        assert breakdown["exact_skill_match"] == 0.85
        assert breakdown["semantic_skill_match"] == 0.75
        assert breakdown["document_similarity"] == 0.80
        assert breakdown["experience_years_fit"] == 0.90
        assert breakdown["education_match"] == 1.0
        assert breakdown["recency_score"] == 0.70
        # total and knockout fields are not in ScoreBreakdown
        assert "total" not in breakdown
        assert "knockout_triggered" not in breakdown

    def test_default_knockout_values(self) -> None:
        """Test default values for knockout fields."""
        scores = AlgorithmicScores(
            exact_skill_match=0.5,
            semantic_skill_match=0.5,
            document_similarity=0.5,
            experience_years_fit=0.5,
            education_match=0.5,
            recency_score=0.5,
            total=0.5,
        )

        assert scores.knockout_triggered is False
        assert scores.knockout_reason is None


class TestLLMAssessmentOutput:
    """Tests for LLMAssessmentOutput Pydantic model."""

    def test_create_valid_assessment(self) -> None:
        """Test creating a valid LLMAssessmentOutput."""
        assessment = LLMAssessmentOutput(
            transferable_skills=["Leadership", "Problem-solving"],
            contextual_strengths=["Strong career progression"],
            concerns=["Job hopping pattern"],
            adjustment=0.05,
            adjustment_rationale="Transferable skills compensate for gaps",
        )

        assert len(assessment.transferable_skills) == 2
        assert assessment.adjustment == 0.05

    def test_default_values(self) -> None:
        """Test default values for LLMAssessmentOutput."""
        assessment = LLMAssessmentOutput()

        assert assessment.transferable_skills == []
        assert assessment.contextual_strengths == []
        assert assessment.concerns == []
        assert assessment.adjustment == 0.0
        assert assessment.adjustment_rationale == ""

    def test_adjustment_boundary_positive_max(self) -> None:
        """Test maximum positive adjustment (0.1)."""
        assessment = LLMAssessmentOutput(adjustment=0.1)

        assert assessment.adjustment == 0.1

    def test_adjustment_boundary_negative_max(self) -> None:
        """Test maximum negative adjustment (-0.1)."""
        assessment = LLMAssessmentOutput(adjustment=-0.1)

        assert assessment.adjustment == -0.1

    def test_adjustment_exceeds_positive_max(self) -> None:
        """Test that adjustment > 0.1 raises validation error."""
        with pytest.raises(ValidationError):
            LLMAssessmentOutput(adjustment=0.15)

    def test_adjustment_exceeds_negative_max(self) -> None:
        """Test that adjustment < -0.1 raises validation error."""
        with pytest.raises(ValidationError):
            LLMAssessmentOutput(adjustment=-0.15)

    def test_partial_initialization(self) -> None:
        """Test partial initialization with some fields."""
        assessment = LLMAssessmentOutput(
            transferable_skills=["Python expertise"],
            adjustment=0.03,
        )

        assert assessment.transferable_skills == ["Python expertise"]
        assert assessment.adjustment == 0.03
        assert assessment.concerns == []  # default


class TestHybridMatchResult:
    """Tests for HybridMatchResult TypedDict."""

    def test_create_valid_result(self) -> None:
        """Test creating a valid HybridMatchResult."""
        breakdown: ScoreBreakdown = {
            "exact_skill_match": 0.85,
            "semantic_skill_match": 0.75,
            "document_similarity": 0.80,
            "experience_years_fit": 0.90,
            "education_match": 1.0,
            "recency_score": 0.70,
        }

        result: HybridMatchResult = {
            "strong_matches": ["Python", "AWS"],
            "partial_matches": ["Cloud experience"],
            "gaps": ["Terraform"],
            "transferable_skills": ["Leadership"],
            "relevance_score": 0.82,
            "score_breakdown": breakdown,
            "algorithmic_score": 0.80,
            "llm_adjustment": 0.02,
            "knockout_triggered": False,
            "knockout_reason": None,
            "scoring_method": "hybrid",
        }

        assert result["relevance_score"] == 0.82
        assert result["algorithmic_score"] == 0.80
        assert result["llm_adjustment"] == 0.02
        assert result["scoring_method"] == "hybrid"

    def test_result_with_knockout(self) -> None:
        """Test HybridMatchResult with knockout triggered."""
        breakdown: ScoreBreakdown = {
            "exact_skill_match": 0.0,
            "semantic_skill_match": 0.0,
            "document_similarity": 0.0,
            "experience_years_fit": 0.0,
            "education_match": 0.0,
            "recency_score": 0.0,
        }

        result: HybridMatchResult = {
            "strong_matches": [],
            "partial_matches": [],
            "gaps": ["Missing required skills: Python"],
            "transferable_skills": [],
            "relevance_score": 0.0,
            "score_breakdown": breakdown,
            "algorithmic_score": 0.0,
            "llm_adjustment": 0.0,
            "knockout_triggered": True,
            "knockout_reason": "Missing required skills: Python",
            "scoring_method": "hybrid",
        }

        assert result["knockout_triggered"] is True
        assert result["relevance_score"] == 0.0

    def test_result_llm_only_method(self) -> None:
        """Test HybridMatchResult with llm_only scoring method."""
        breakdown: ScoreBreakdown = {
            "exact_skill_match": 0.0,
            "semantic_skill_match": 0.0,
            "document_similarity": 0.0,
            "experience_years_fit": 0.0,
            "education_match": 0.0,
            "recency_score": 0.0,
        }

        result: HybridMatchResult = {
            "strong_matches": ["Python"],
            "partial_matches": [],
            "gaps": [],
            "transferable_skills": [],
            "relevance_score": 0.75,
            "score_breakdown": breakdown,
            "algorithmic_score": 0.0,
            "llm_adjustment": 0.75,
            "knockout_triggered": False,
            "knockout_reason": None,
            "scoring_method": "llm_only",
        }

        assert result["scoring_method"] == "llm_only"

    def test_result_score_breakdown_access(self) -> None:
        """Test accessing score breakdown from result."""
        breakdown: ScoreBreakdown = {
            "exact_skill_match": 0.85,
            "semantic_skill_match": 0.75,
            "document_similarity": 0.80,
            "experience_years_fit": 0.90,
            "education_match": 1.0,
            "recency_score": 0.70,
        }

        result: HybridMatchResult = {
            "strong_matches": [],
            "partial_matches": [],
            "gaps": [],
            "transferable_skills": [],
            "relevance_score": 0.82,
            "score_breakdown": breakdown,
            "algorithmic_score": 0.80,
            "llm_adjustment": 0.02,
            "knockout_triggered": False,
            "knockout_reason": None,
            "scoring_method": "hybrid",
        }

        assert result["score_breakdown"]["exact_skill_match"] == 0.85
        assert result["score_breakdown"]["recency_score"] == 0.70
