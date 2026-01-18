"""Tests for the MatchAnalyzer processor."""

from unittest.mock import MagicMock, patch

import pytest

from cv_warlock.models.cv import ContactInfo, CVData, Education, Experience
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.models.state import MatchAnalysis, TailoringPlan
from cv_warlock.processors.matcher import (
    MatchAnalysisOutput,
    MatchAnalyzer,
    TailoringPlanOutput,
)


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLM provider."""
    return MagicMock()


@pytest.fixture
def sample_cv_data() -> CVData:
    """Create sample CV data for testing."""
    return CVData(
        contact=ContactInfo(name="John Doe", email="john@example.com"),
        summary="Experienced software engineer with 10+ years building scalable systems.",
        experiences=[
            Experience(
                title="Senior Software Engineer",
                company="Tech Corp",
                start_date="Jan 2020",
                end_date="Present",
                description="Led development of cloud-native applications",
                achievements=["Led team of 5", "Reduced deployment time by 50%"],
                skills_used=["Python", "AWS", "Docker"],
            )
        ],
        education=[
            Education(
                degree="MS Computer Science",
                institution="Stanford",
                graduation_date="2015",
            )
        ],
        skills=["Python", "AWS", "Docker", "Kubernetes", "PostgreSQL"],
    )


@pytest.fixture
def sample_job_requirements() -> JobRequirements:
    """Create sample job requirements for testing."""
    return JobRequirements(
        job_title="Senior Software Engineer",
        company="Acme Inc",
        required_skills=["Python", "AWS", "Docker"],
        preferred_skills=["Kubernetes", "Terraform"],
        required_experience_years=5,
        required_education="Bachelor's degree",
        seniority_level="senior",
        responsibilities=["Design systems", "Lead projects"],
    )


class TestMatchAnalysisOutput:
    """Tests for the MatchAnalysisOutput Pydantic model."""

    def test_create_valid_output(self) -> None:
        """Test creating a valid MatchAnalysisOutput."""
        output = MatchAnalysisOutput(
            strong_matches=["Python", "AWS"],
            partial_matches=["Cloud experience"],
            gaps=["Terraform"],
            transferable_skills=["Redis"],
            relevance_score=0.85,
        )

        assert output.strong_matches == ["Python", "AWS"]
        assert output.relevance_score == 0.85

    def test_relevance_score_bounds(self) -> None:
        """Test that relevance_score is bounded between 0 and 1."""
        # Valid boundary values
        output_zero = MatchAnalysisOutput(
            strong_matches=[],
            partial_matches=[],
            gaps=[],
            transferable_skills=[],
            relevance_score=0.0,
        )
        assert output_zero.relevance_score == 0.0

        output_one = MatchAnalysisOutput(
            strong_matches=[],
            partial_matches=[],
            gaps=[],
            transferable_skills=[],
            relevance_score=1.0,
        )
        assert output_one.relevance_score == 1.0

    def test_relevance_score_invalid_low(self) -> None:
        """Test that relevance_score < 0 raises validation error."""
        with pytest.raises(ValueError):
            MatchAnalysisOutput(
                strong_matches=[],
                partial_matches=[],
                gaps=[],
                transferable_skills=[],
                relevance_score=-0.1,
            )

    def test_relevance_score_invalid_high(self) -> None:
        """Test that relevance_score > 1 raises validation error."""
        with pytest.raises(ValueError):
            MatchAnalysisOutput(
                strong_matches=[],
                partial_matches=[],
                gaps=[],
                transferable_skills=[],
                relevance_score=1.1,
            )


class TestTailoringPlanOutput:
    """Tests for the TailoringPlanOutput Pydantic model."""

    def test_create_valid_output(self) -> None:
        """Test creating a valid TailoringPlanOutput."""
        output = TailoringPlanOutput(
            summary_focus=["Cloud-native", "Leadership"],
            experiences_to_emphasize=["Senior Engineer role"],
            skills_to_highlight=["Python", "AWS"],
            achievements_to_feature=["Led team of 5"],
            keywords_to_incorporate=["microservices"],
            sections_to_reorder=["summary", "experience"],
        )

        assert output.summary_focus == ["Cloud-native", "Leadership"]
        assert len(output.skills_to_highlight) == 2

    def test_empty_lists_valid(self) -> None:
        """Test that empty lists are valid for TailoringPlanOutput."""
        output = TailoringPlanOutput(
            summary_focus=[],
            experiences_to_emphasize=[],
            skills_to_highlight=[],
            achievements_to_feature=[],
            keywords_to_incorporate=[],
            sections_to_reorder=[],
        )

        assert output.summary_focus == []


class TestMatchAnalyzer:
    """Tests for the MatchAnalyzer class."""

    def test_init_sets_provider_and_prompts(self, mock_provider: MagicMock) -> None:
        """Test that __init__ sets up the provider and prompts."""
        analyzer = MatchAnalyzer(mock_provider)

        assert analyzer.llm_provider is mock_provider
        assert analyzer.analysis_prompt is not None
        assert analyzer.plan_prompt is not None

    def test_analyze_match_returns_match_analysis(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that analyze_match returns a MatchAnalysis."""
        expected_result: MatchAnalysis = {
            "strong_matches": ["Python", "AWS", "Docker"],
            "partial_matches": ["Cloud experience"],
            "gaps": ["Terraform"],
            "transferable_skills": ["Kubernetes"],
            "relevance_score": 0.85,
        }

        with patch.object(MatchAnalyzer, "analyze_match", return_value=expected_result):
            analyzer = MatchAnalyzer(mock_provider)
            result = analyzer.analyze_match(sample_cv_data, sample_job_requirements)

        # Verify the structure of MatchAnalysis
        assert "strong_matches" in result
        assert "partial_matches" in result
        assert "gaps" in result
        assert "transferable_skills" in result
        assert "relevance_score" in result
        assert result["relevance_score"] == 0.85

    def test_create_tailoring_plan_returns_tailoring_plan(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that create_tailoring_plan returns a TailoringPlan."""
        sample_match: MatchAnalysis = {
            "strong_matches": ["Python"],
            "partial_matches": [],
            "gaps": [],
            "transferable_skills": [],
            "relevance_score": 0.9,
        }

        expected_plan: TailoringPlan = {
            "summary_focus": ["Technical leadership"],
            "experiences_to_emphasize": ["Current role"],
            "skills_to_highlight": ["Python"],
            "achievements_to_feature": ["Revenue growth"],
            "keywords_to_incorporate": ["scalability"],
            "sections_to_reorder": ["experience", "skills"],
        }

        with patch.object(MatchAnalyzer, "create_tailoring_plan", return_value=expected_plan):
            analyzer = MatchAnalyzer(mock_provider)
            result = analyzer.create_tailoring_plan(
                sample_cv_data, sample_job_requirements, sample_match
            )

        # Verify the structure of TailoringPlan
        assert "summary_focus" in result
        assert "experiences_to_emphasize" in result
        assert "skills_to_highlight" in result
        assert "achievements_to_feature" in result
        assert "keywords_to_incorporate" in result
        assert "sections_to_reorder" in result


class TestMatchAnalyzerWorkflow:
    """Tests for MatchAnalyzer workflow scenarios."""

    def test_full_analysis_workflow(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test a complete analysis workflow: analyze_match then create_tailoring_plan."""
        match_result: MatchAnalysis = {
            "strong_matches": ["Python", "AWS"],
            "partial_matches": ["Docker"],
            "gaps": ["Terraform"],
            "transferable_skills": ["Kubernetes"],
            "relevance_score": 0.82,
        }

        plan_result: TailoringPlan = {
            "summary_focus": ["Cloud-native expertise"],
            "experiences_to_emphasize": ["Tech Corp Senior Engineer"],
            "skills_to_highlight": ["Python", "AWS", "Docker"],
            "achievements_to_feature": ["Led team of 5", "50% deployment improvement"],
            "keywords_to_incorporate": ["microservices", "scalability"],
            "sections_to_reorder": ["summary", "experience", "skills"],
        }

        with patch.object(MatchAnalyzer, "analyze_match", return_value=match_result):
            with patch.object(MatchAnalyzer, "create_tailoring_plan", return_value=plan_result):
                analyzer = MatchAnalyzer(mock_provider)

                # Step 1: Analyze match
                match = analyzer.analyze_match(sample_cv_data, sample_job_requirements)

                # Step 2: Create tailoring plan
                plan = analyzer.create_tailoring_plan(
                    sample_cv_data, sample_job_requirements, match
                )

        # Verify both steps completed
        assert match["relevance_score"] == 0.82
        assert len(match["strong_matches"]) == 2
        assert plan["summary_focus"] == ["Cloud-native expertise"]
        assert len(plan["achievements_to_feature"]) == 2
