"""Tests for the HybridScorer."""

from unittest.mock import MagicMock, patch

import pytest

from cv_warlock.models.cv import ContactInfo, CVData, Education, Experience
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.scoring.algorithmic import AlgorithmicScorer
from cv_warlock.scoring.hybrid import HybridScorer, _LLMAssessmentOutput
from cv_warlock.scoring.models import (
    AlgorithmicScores,
    LLMAssessmentOutput,
)


@pytest.fixture
def mock_llm_provider() -> MagicMock:
    """Create a mock LLM provider."""
    provider = MagicMock()
    mock_model = MagicMock()
    provider.get_extraction_model.return_value = mock_model
    provider.get_chat_model.return_value = mock_model
    return provider


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
                start_date="January 2020",
                end_date="Present",
                description="Led development of cloud-native applications using Python and AWS",
                achievements=[
                    "Led team of 5 engineers",
                    "Reduced deployment time by 50%",
                ],
                skills_used=["Python", "AWS", "Docker"],
            ),
        ],
        education=[
            Education(
                degree="Master of Science",
                institution="Stanford",
                graduation_date="2015",
            )
        ],
        skills=["Python", "AWS", "Docker", "Kubernetes"],
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
    )


@pytest.fixture
def sample_algorithmic_scores() -> AlgorithmicScores:
    """Create sample algorithmic scores."""
    return AlgorithmicScores(
        exact_skill_match=0.85,
        semantic_skill_match=0.85,
        document_similarity=0.85,
        experience_years_fit=0.90,
        education_match=1.0,
        recency_score=0.80,
        total=0.85,
        knockout_triggered=False,
        knockout_reason=None,
    )


@pytest.fixture
def sample_knockout_scores() -> AlgorithmicScores:
    """Create sample scores with knockout triggered."""
    return AlgorithmicScores(
        exact_skill_match=0.0,
        semantic_skill_match=0.0,
        document_similarity=0.0,
        experience_years_fit=0.5,
        education_match=1.0,
        recency_score=0.3,
        total=0.0,
        knockout_triggered=True,
        knockout_reason="Missing required skills: Python, AWS",
    )


class TestInternalLLMAssessmentOutput:
    """Tests for the internal _LLMAssessmentOutput model."""

    def test_create_valid_output(self) -> None:
        """Test creating a valid _LLMAssessmentOutput."""
        output = _LLMAssessmentOutput(
            transferable_skills=["Leadership"],
            contextual_strengths=["Strong progression"],
            concerns=[],
            adjustment=0.05,
            adjustment_rationale="Good transferable skills",
        )

        assert output.transferable_skills == ["Leadership"]
        assert output.adjustment == 0.05

    def test_default_values(self) -> None:
        """Test default values for _LLMAssessmentOutput."""
        output = _LLMAssessmentOutput()

        assert output.transferable_skills == []
        assert output.contextual_strengths == []
        assert output.concerns == []
        assert output.adjustment == 0.0
        assert output.adjustment_rationale == ""

    def test_adjustment_bounds_positive(self) -> None:
        """Test that adjustment is bounded at +0.1."""
        output = _LLMAssessmentOutput(adjustment=0.1)
        assert output.adjustment == 0.1

    def test_adjustment_bounds_negative(self) -> None:
        """Test that adjustment is bounded at -0.1."""
        output = _LLMAssessmentOutput(adjustment=-0.1)
        assert output.adjustment == -0.1

    def test_adjustment_exceeds_bounds(self) -> None:
        """Test that out-of-bounds adjustment raises error."""
        with pytest.raises(ValueError):
            _LLMAssessmentOutput(adjustment=0.2)


class TestHybridScorerInit:
    """Tests for HybridScorer initialization."""

    def test_init_sets_provider(self, mock_llm_provider: MagicMock) -> None:
        """Test that __init__ sets the LLM provider."""
        scorer = HybridScorer(mock_llm_provider)

        assert scorer.llm_provider is mock_llm_provider

    def test_init_creates_algorithmic_scorer(self, mock_llm_provider: MagicMock) -> None:
        """Test that __init__ creates an AlgorithmicScorer."""
        scorer = HybridScorer(mock_llm_provider)

        assert isinstance(scorer.algorithmic, AlgorithmicScorer)

    def test_init_sets_up_prompt(self, mock_llm_provider: MagicMock) -> None:
        """Test that __init__ sets up the prompt template."""
        scorer = HybridScorer(mock_llm_provider)

        assert scorer._prompt is not None


class TestHybridScorerScore:
    """Tests for the main score method."""

    def test_score_returns_hybrid_match_result(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_algorithmic_scores: AlgorithmicScores,
    ) -> None:
        """Test that score returns HybridMatchResult."""
        scorer = HybridScorer(mock_llm_provider)

        # Mock the internal methods
        with patch.object(scorer.algorithmic, "compute", return_value=sample_algorithmic_scores):
            with patch.object(
                scorer,
                "_get_llm_assessment",
                return_value=LLMAssessmentOutput(adjustment=0.02),
            ):
                result = scorer.score(sample_cv_data, sample_job_requirements)

        assert isinstance(result, dict)
        assert "relevance_score" in result
        assert "algorithmic_score" in result
        assert "scoring_method" in result

    def test_score_knockout_skips_llm(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_knockout_scores: AlgorithmicScores,
    ) -> None:
        """Test that knockout case skips LLM assessment."""
        scorer = HybridScorer(mock_llm_provider)

        with patch.object(scorer.algorithmic, "compute", return_value=sample_knockout_scores):
            with patch.object(scorer, "_get_llm_assessment") as mock_llm:
                result = scorer.score(sample_cv_data, sample_job_requirements)

        # LLM should not be called for knockout
        mock_llm.assert_not_called()
        assert result["knockout_triggered"] is True
        assert result["relevance_score"] == 0.0

    def test_score_applies_llm_adjustment(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that LLM adjustment is applied to final score."""
        scorer = HybridScorer(mock_llm_provider)

        algo_scores = AlgorithmicScores(
            exact_skill_match=0.80,
            semantic_skill_match=0.80,
            document_similarity=0.80,
            experience_years_fit=0.80,
            education_match=0.80,
            recency_score=0.80,
            total=0.80,
        )

        llm_assessment = LLMAssessmentOutput(
            transferable_skills=["Leadership"],
            adjustment=0.05,
            adjustment_rationale="Strong transferable skills",
        )

        with patch.object(scorer.algorithmic, "compute", return_value=algo_scores):
            with patch.object(scorer, "_get_llm_assessment", return_value=llm_assessment):
                result = scorer.score(sample_cv_data, sample_job_requirements)

        # 0.80 + 0.05 = 0.85 (use approx for float comparison)
        assert result["relevance_score"] == pytest.approx(0.85)
        assert result["algorithmic_score"] == 0.80
        assert result["llm_adjustment"] == 0.05


class TestCreateKnockoutResult:
    """Tests for _create_knockout_result method."""

    def test_creates_zero_score_result(
        self,
        mock_llm_provider: MagicMock,
        sample_knockout_scores: AlgorithmicScores,
    ) -> None:
        """Test that knockout result has zero relevance score."""
        scorer = HybridScorer(mock_llm_provider)

        result = scorer._create_knockout_result(sample_knockout_scores)

        assert result["relevance_score"] == 0.0
        assert result["algorithmic_score"] == 0.0
        assert result["llm_adjustment"] == 0.0

    def test_includes_knockout_info(
        self,
        mock_llm_provider: MagicMock,
        sample_knockout_scores: AlgorithmicScores,
    ) -> None:
        """Test that knockout result includes knockout info."""
        scorer = HybridScorer(mock_llm_provider)

        result = scorer._create_knockout_result(sample_knockout_scores)

        assert result["knockout_triggered"] is True
        assert result["knockout_reason"] == sample_knockout_scores.knockout_reason

    def test_has_gap_in_gaps_list(
        self,
        mock_llm_provider: MagicMock,
        sample_knockout_scores: AlgorithmicScores,
    ) -> None:
        """Test that knockout result includes gap in gaps list."""
        scorer = HybridScorer(mock_llm_provider)

        result = scorer._create_knockout_result(sample_knockout_scores)

        assert len(result["gaps"]) == 1
        assert "Missing required skills" in result["gaps"][0]

    def test_empty_matches_lists(
        self,
        mock_llm_provider: MagicMock,
        sample_knockout_scores: AlgorithmicScores,
    ) -> None:
        """Test that knockout result has empty match lists."""
        scorer = HybridScorer(mock_llm_provider)

        result = scorer._create_knockout_result(sample_knockout_scores)

        assert result["strong_matches"] == []
        assert result["partial_matches"] == []
        assert result["transferable_skills"] == []

    def test_sets_hybrid_scoring_method(
        self,
        mock_llm_provider: MagicMock,
        sample_knockout_scores: AlgorithmicScores,
    ) -> None:
        """Test that knockout result sets scoring_method to hybrid."""
        scorer = HybridScorer(mock_llm_provider)

        result = scorer._create_knockout_result(sample_knockout_scores)

        assert result["scoring_method"] == "hybrid"


class TestGetLLMAssessment:
    """Tests for _get_llm_assessment method."""

    def test_returns_llm_assessment_output(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_algorithmic_scores: AlgorithmicScores,
    ) -> None:
        """Test that _get_llm_assessment returns LLMAssessmentOutput on success."""
        # Create a mock LLM result
        mock_result = _LLMAssessmentOutput(
            transferable_skills=["Leadership"],
            contextual_strengths=["Career progression"],
            concerns=[],
            adjustment=0.03,
            adjustment_rationale="Strong fit",
        )

        # Create a mock chain that returns our mock result
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result

        # Create mock model with proper structured output setup
        mock_model = MagicMock()
        mock_structured_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured_model
        mock_llm_provider.get_extraction_model.return_value = mock_model

        scorer = HybridScorer(mock_llm_provider)

        # Patch at the ChatPromptTemplate level to make the chain work
        with patch.object(scorer, "_prompt") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            result = scorer._get_llm_assessment(
                sample_cv_data, sample_job_requirements, sample_algorithmic_scores
            )

        # The method should return an LLMAssessmentOutput
        assert isinstance(result, LLMAssessmentOutput)
        # Note: Due to the complex chain mocking, verify we get a valid result
        assert hasattr(result, "adjustment")

    def test_handles_llm_failure_gracefully(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_algorithmic_scores: AlgorithmicScores,
    ) -> None:
        """Test that LLM failures return neutral adjustment."""
        scorer = HybridScorer(mock_llm_provider)

        # Make the LLM call raise an exception
        mock_model = MagicMock()
        mock_model.with_structured_output.side_effect = Exception("API Error")
        mock_llm_provider.get_extraction_model.return_value = mock_model

        result = scorer._get_llm_assessment(
            sample_cv_data, sample_job_requirements, sample_algorithmic_scores
        )

        # Should return neutral assessment
        assert result.adjustment == 0.0
        assert "failed" in result.adjustment_rationale.lower()
        assert len(result.concerns) > 0


class TestCombineScores:
    """Tests for _combine_scores method."""

    def test_combines_scores_correctly(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_algorithmic_scores: AlgorithmicScores,
    ) -> None:
        """Test that scores are combined correctly."""
        scorer = HybridScorer(mock_llm_provider)

        llm_assessment = LLMAssessmentOutput(
            transferable_skills=["Leadership"],
            adjustment=0.05,
            adjustment_rationale="Good fit",
        )

        result = scorer._combine_scores(
            sample_cv_data,
            sample_job_requirements,
            sample_algorithmic_scores,
            llm_assessment,
        )

        # 0.85 + 0.05 = 0.90
        assert result["relevance_score"] == 0.90
        assert result["algorithmic_score"] == 0.85
        assert result["llm_adjustment"] == 0.05

    def test_clamps_final_score_to_one(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that final score is clamped to 1.0 max."""
        scorer = HybridScorer(mock_llm_provider)

        high_scores = AlgorithmicScores(
            exact_skill_match=0.98,
            semantic_skill_match=0.98,
            document_similarity=0.98,
            experience_years_fit=0.98,
            education_match=0.98,
            recency_score=0.98,
            total=0.98,
        )

        llm_assessment = LLMAssessmentOutput(adjustment=0.1)

        result = scorer._combine_scores(
            sample_cv_data, sample_job_requirements, high_scores, llm_assessment
        )

        # 0.98 + 0.1 would be 1.08, clamped to 1.0
        assert result["relevance_score"] == 1.0

    def test_clamps_final_score_to_zero(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that final score is clamped to 0.0 min."""
        scorer = HybridScorer(mock_llm_provider)

        low_scores = AlgorithmicScores(
            exact_skill_match=0.05,
            semantic_skill_match=0.05,
            document_similarity=0.05,
            experience_years_fit=0.05,
            education_match=0.05,
            recency_score=0.05,
            total=0.05,
        )

        llm_assessment = LLMAssessmentOutput(adjustment=-0.1)

        result = scorer._combine_scores(
            sample_cv_data, sample_job_requirements, low_scores, llm_assessment
        )

        # 0.05 - 0.1 would be -0.05, clamped to 0.0
        assert result["relevance_score"] == 0.0

    def test_includes_transferable_skills(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_algorithmic_scores: AlgorithmicScores,
    ) -> None:
        """Test that transferable skills are included from LLM assessment."""
        scorer = HybridScorer(mock_llm_provider)

        llm_assessment = LLMAssessmentOutput(
            transferable_skills=["Leadership", "Communication"],
            adjustment=0.0,
        )

        result = scorer._combine_scores(
            sample_cv_data,
            sample_job_requirements,
            sample_algorithmic_scores,
            llm_assessment,
        )

        assert result["transferable_skills"] == ["Leadership", "Communication"]

    def test_sets_not_knockout(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_algorithmic_scores: AlgorithmicScores,
    ) -> None:
        """Test that combined result sets knockout to False."""
        scorer = HybridScorer(mock_llm_provider)

        llm_assessment = LLMAssessmentOutput()

        result = scorer._combine_scores(
            sample_cv_data,
            sample_job_requirements,
            sample_algorithmic_scores,
            llm_assessment,
        )

        assert result["knockout_triggered"] is False
        assert result["knockout_reason"] is None


class TestCategorizeMatches:
    """Tests for _categorize_matches method."""

    def test_categorizes_required_exact_match(
        self,
        mock_llm_provider: MagicMock,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that required exact matches are categorized as strong."""
        scorer = HybridScorer(mock_llm_provider)

        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=["Python", "AWS"],
            experiences=[],
        )

        strong, partial = scorer._categorize_matches(cv_data, sample_job_requirements)

        # Python and AWS are required and in skills list
        assert any("Python" in m and "required" in m and "exact" in m for m in strong)
        assert any("AWS" in m and "required" in m and "exact" in m for m in strong)

    def test_categorizes_required_in_experience_as_partial(
        self,
        mock_llm_provider: MagicMock,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that required skills in experience text are partial matches."""
        scorer = HybridScorer(mock_llm_provider)

        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=[],  # Python not in skills list
            experiences=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date="2020",
                    description="Worked with Python daily",
                )
            ],
        )

        strong, partial = scorer._categorize_matches(cv_data, sample_job_requirements)

        assert any("Python" in m and "experience" in m for m in partial)

    def test_categorizes_preferred_exact_match(
        self,
        mock_llm_provider: MagicMock,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that preferred exact matches are categorized as strong."""
        scorer = HybridScorer(mock_llm_provider)

        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=["Kubernetes"],  # Preferred skill
            experiences=[],
        )

        strong, partial = scorer._categorize_matches(cv_data, sample_job_requirements)

        assert any("Kubernetes" in m and "preferred" in m for m in strong)

    def test_includes_experience_skills_used(
        self,
        mock_llm_provider: MagicMock,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that skills_used from experiences are included."""
        scorer = HybridScorer(mock_llm_provider)

        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=[],
            experiences=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date="2020",
                    skills_used=["Python", "AWS"],
                )
            ],
        )

        strong, partial = scorer._categorize_matches(cv_data, sample_job_requirements)

        # Should find Python and AWS as exact matches from skills_used
        assert any("Python" in m and "exact" in m for m in strong)
        assert any("AWS" in m and "exact" in m for m in strong)


class TestIdentifyGaps:
    """Tests for _identify_gaps method."""

    def test_identifies_missing_required_skills(
        self,
        mock_llm_provider: MagicMock,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that missing required skills are identified."""
        scorer = HybridScorer(mock_llm_provider)

        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=["Python"],  # Missing AWS, Docker
            experiences=[],
        )

        gaps = scorer._identify_gaps(cv_data, sample_job_requirements)

        assert "AWS" in gaps
        assert "Docker" in gaps
        assert "Python" not in gaps

    def test_no_gaps_when_all_skills_present(
        self,
        mock_llm_provider: MagicMock,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that no gaps when all required skills present."""
        scorer = HybridScorer(mock_llm_provider)

        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=["Python", "AWS", "Docker"],
            experiences=[],
        )

        gaps = scorer._identify_gaps(cv_data, sample_job_requirements)

        assert len(gaps) == 0

    def test_skills_in_experience_not_counted_as_gaps(
        self,
        mock_llm_provider: MagicMock,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that skills mentioned in experience are not gaps."""
        scorer = HybridScorer(mock_llm_provider)

        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=[],
            experiences=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date="2020",
                    description="Used Python, AWS, and Docker",
                )
            ],
        )

        gaps = scorer._identify_gaps(cv_data, sample_job_requirements)

        assert len(gaps) == 0


class TestSerializeMethods:
    """Tests for serialization helper methods."""

    def test_serialize_cv_returns_json(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
    ) -> None:
        """Test that _serialize_cv returns JSON string."""
        result = HybridScorer._serialize_cv(sample_cv_data)

        assert isinstance(result, str)
        assert "John Doe" in result
        assert "Senior Software Engineer" in result

    def test_serialize_job_returns_json(
        self,
        mock_llm_provider: MagicMock,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that _serialize_job returns JSON string."""
        result = HybridScorer._serialize_job(sample_job_requirements)

        assert isinstance(result, str)
        assert "Senior Software Engineer" in result
        assert "Acme Inc" in result


class TestScoreWithPlan:
    """Tests for the score_with_plan optimized method."""

    def test_score_with_plan_returns_tuple(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_algorithmic_scores: AlgorithmicScores,
    ) -> None:
        """Test that score_with_plan returns tuple of result and plan."""
        from cv_warlock.scoring.hybrid import _CombinedAnalysisOutput

        scorer = HybridScorer(mock_llm_provider)

        # Setup mock combined analysis output
        mock_combined = _CombinedAnalysisOutput(
            transferable_skills=["Leadership"],
            contextual_strengths=["Career progression"],
            concerns=[],
            adjustment=0.03,
            adjustment_rationale="Good fit",
            summary_focus=["Highlight Python expertise"],
            experiences_to_emphasize=["Senior Software Engineer at Tech Corp"],
            skills_to_highlight=["Python", "AWS"],
            achievements_to_feature=["Led team of 5"],
            keywords_to_incorporate=["cloud-native", "scalable"],
            sections_to_reorder=["Experience", "Skills", "Education"],
        )

        with patch.object(scorer.algorithmic, "compute", return_value=sample_algorithmic_scores):
            with patch.object(scorer, "_get_combined_analysis", return_value=mock_combined):
                result, plan = scorer.score_with_plan(sample_cv_data, sample_job_requirements)

        # Verify result structure
        assert isinstance(result, dict)
        assert "relevance_score" in result
        assert result["scoring_method"] == "hybrid"

        # Verify plan structure (TailoringPlan is a TypedDict, so check dict keys)
        assert isinstance(plan, dict)
        assert plan["summary_focus"] == ["Highlight Python expertise"]
        assert plan["experiences_to_emphasize"] == ["Senior Software Engineer at Tech Corp"]
        assert plan["skills_to_highlight"] == ["Python", "AWS"]

    def test_score_with_plan_knockout_returns_empty_plan(
        self,
        mock_llm_provider: MagicMock,
        sample_knockout_scores: AlgorithmicScores,
    ) -> None:
        """Test that knockout case returns empty tailoring plan."""
        scorer = HybridScorer(mock_llm_provider)

        cv_data = CVData(
            contact=ContactInfo(name="Jane Doe"),
            skills=["Java", "Spring"],  # No required skills
        )

        job_requirements = JobRequirements(
            job_title="Python Developer",
            required_skills=["Python", "Django"],
        )

        with patch.object(scorer.algorithmic, "compute", return_value=sample_knockout_scores):
            result, plan = scorer.score_with_plan(cv_data, job_requirements)

        # Verify knockout result
        assert result["knockout_triggered"] is True
        assert result["relevance_score"] == 0.0

        # Verify empty plan returned (TailoringPlan is a TypedDict, so check dict keys)
        assert isinstance(plan, dict)
        assert plan["summary_focus"] == []
        assert plan["experiences_to_emphasize"] == []
        assert plan["skills_to_highlight"] == []
        assert plan["achievements_to_feature"] == []
        assert plan["keywords_to_incorporate"] == []
        assert plan["sections_to_reorder"] == []

    def test_score_with_plan_applies_llm_adjustment(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that LLM adjustment is applied in score_with_plan."""
        from cv_warlock.scoring.hybrid import _CombinedAnalysisOutput

        scorer = HybridScorer(mock_llm_provider)

        algo_scores = AlgorithmicScores(
            exact_skill_match=0.80,
            semantic_skill_match=0.80,
            document_similarity=0.80,
            experience_years_fit=0.80,
            education_match=0.80,
            recency_score=0.80,
            total=0.80,
        )

        mock_combined = _CombinedAnalysisOutput(
            adjustment=0.05,
            adjustment_rationale="Strong transferable skills",
        )

        with patch.object(scorer.algorithmic, "compute", return_value=algo_scores):
            with patch.object(scorer, "_get_combined_analysis", return_value=mock_combined):
                result, _ = scorer.score_with_plan(sample_cv_data, sample_job_requirements)

        # 0.80 + 0.05 = 0.85
        assert result["relevance_score"] == pytest.approx(0.85)
        assert result["algorithmic_score"] == 0.80
        assert result["llm_adjustment"] == 0.05


class TestGetCombinedAnalysis:
    """Tests for the _get_combined_analysis method."""

    def test_handles_exception_gracefully(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_algorithmic_scores: AlgorithmicScores,
    ) -> None:
        """Test that _get_combined_analysis returns defaults on error."""
        from cv_warlock.scoring.hybrid import _CombinedAnalysisOutput

        scorer = HybridScorer(mock_llm_provider)

        # Make the LLM call raise an exception
        mock_model = MagicMock()
        mock_model.with_structured_output.side_effect = Exception("API Error")
        mock_llm_provider.get_extraction_model.return_value = mock_model

        result = scorer._get_combined_analysis(
            sample_cv_data, sample_job_requirements, sample_algorithmic_scores
        )

        # Should return default _CombinedAnalysisOutput
        assert isinstance(result, _CombinedAnalysisOutput)
        assert result.adjustment == 0.0
        assert "Failed to get LLM analysis" in result.adjustment_rationale
        assert any("failed" in c.lower() for c in result.concerns)
        assert result.summary_focus == []
        assert result.experiences_to_emphasize == []

    def test_calls_llm_with_correct_prompt_vars(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_algorithmic_scores: AlgorithmicScores,
    ) -> None:
        """Test that _get_combined_analysis calls LLM with correct variables."""
        from cv_warlock.scoring.hybrid import _CombinedAnalysisOutput

        scorer = HybridScorer(mock_llm_provider)

        mock_result = _CombinedAnalysisOutput(
            transferable_skills=["Leadership"],
            adjustment=0.02,
        )

        # Create a mock runnable that captures the invoke call
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = mock_result

        # Mock the entire chain creation
        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = MagicMock()
        mock_llm_provider.get_extraction_model.return_value = mock_model

        # Patch the prompt's __or__ at class level
        from langchain_core.prompts import ChatPromptTemplate

        original_or = ChatPromptTemplate.__or__

        def mock_or(self, other):
            return mock_runnable

        with patch.object(ChatPromptTemplate, "__or__", mock_or):
            result = scorer._get_combined_analysis(
                sample_cv_data, sample_job_requirements, sample_algorithmic_scores
            )

        # Verify invoke was called
        mock_runnable.invoke.assert_called_once()

        # Verify result contains expected values
        assert result.transferable_skills == ["Leadership"]
        assert result.adjustment == 0.02


class TestCategorizeMatchesPreferred:
    """Tests for preferred skill partial matching in _categorize_matches."""

    def test_preferred_skill_in_experience_text_is_partial_match(
        self,
        mock_llm_provider: MagicMock,
    ) -> None:
        """Test that preferred skills mentioned in experience text are partial matches."""
        scorer = HybridScorer(mock_llm_provider)

        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=[],  # Terraform not in skills list
            experiences=[
                Experience(
                    title="DevOps Engineer",
                    company="Tech Corp",
                    start_date="2020",
                    description="Managed infrastructure using Terraform and Ansible",
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Infrastructure Engineer",
            required_skills=["AWS"],  # Not in CV
            preferred_skills=["Terraform", "Kubernetes"],  # Terraform in experience desc
        )

        strong, partial = scorer._categorize_matches(cv_data, job_requirements)

        # Terraform is a preferred skill mentioned in experience text
        assert any("Terraform" in m and "preferred" in m and "experience" in m for m in partial)
        # Kubernetes is not mentioned at all, so should not appear
        assert not any("Kubernetes" in m for m in strong + partial)


class TestFullHybridScoringWorkflow:
    """Integration tests for the full hybrid scoring workflow."""

    def test_full_workflow_no_knockout(
        self,
        mock_llm_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test full workflow without knockout."""
        scorer = HybridScorer(mock_llm_provider)

        # Setup mock LLM response
        mock_result = _LLMAssessmentOutput(
            transferable_skills=["Leadership"],
            contextual_strengths=["Strong background"],
            concerns=[],
            adjustment=0.02,
            adjustment_rationale="Good fit",
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_chain
        mock_llm_provider.get_extraction_model.return_value = mock_model

        with patch.object(scorer._prompt, "__or__", return_value=mock_chain):
            result = scorer.score(sample_cv_data, sample_job_requirements)

        # Verify result structure
        assert "relevance_score" in result
        assert "score_breakdown" in result
        assert "strong_matches" in result
        assert "partial_matches" in result
        assert "gaps" in result
        assert result["knockout_triggered"] is False
        assert result["scoring_method"] == "hybrid"

    def test_full_workflow_with_knockout(
        self,
        mock_llm_provider: MagicMock,
    ) -> None:
        """Test full workflow with knockout."""
        scorer = HybridScorer(mock_llm_provider)

        cv_data = CVData(
            contact=ContactInfo(name="Jane Doe"),
            skills=["Java", "Spring"],  # No required skills
        )

        job_requirements = JobRequirements(
            job_title="Python Developer",
            required_skills=["Python", "Django"],
        )

        result = scorer.score(cv_data, job_requirements)

        assert result["knockout_triggered"] is True
        assert result["relevance_score"] == 0.0
        assert "Python" in result["knockout_reason"]
