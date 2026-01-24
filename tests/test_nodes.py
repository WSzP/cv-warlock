"""Tests for LangGraph workflow nodes."""

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cv_warlock.graph.nodes import (
    STEP_DESCRIPTIONS,
    STEP_DESCRIPTIONS_FAST,
    _end_step,
    _start_step,
    create_nodes,
)
from cv_warlock.models.cv import ContactInfo, CVData, Education, Experience
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.models.state import CVWarlockState, MatchAnalysis, TailoringPlan

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLM provider."""
    return MagicMock()


@pytest.fixture
def sample_cv_data() -> CVData:
    """Create sample CV data."""
    return CVData(
        contact=ContactInfo(name="John Doe", email="john@example.com"),
        summary="Experienced engineer",
        experiences=[
            Experience(
                title="Senior Engineer",
                company="Tech Corp",
                start_date="Jan 2020",
                end_date="Present",
                achievements=["Led team", "Improved systems"],
            )
        ],
        education=[
            Education(
                degree="MS CS",
                institution="Stanford",
                graduation_date="2015",
            )
        ],
        skills=["Python", "AWS", "Docker"],
    )


@pytest.fixture
def sample_job_requirements() -> JobRequirements:
    """Create sample job requirements."""
    return JobRequirements(
        job_title="Senior Software Engineer",
        company="Acme Inc",
        required_skills=["Python", "AWS", "Docker"],
        preferred_skills=["Kubernetes"],
    )


@pytest.fixture
def sample_match_analysis() -> MatchAnalysis:
    """Create sample match analysis."""
    return {
        "strong_matches": ["Python", "AWS"],
        "partial_matches": ["Docker"],
        "gaps": ["Terraform"],
        "transferable_skills": ["Leadership"],
        "relevance_score": 0.85,
    }


@pytest.fixture
def sample_tailoring_plan() -> TailoringPlan:
    """Create sample tailoring plan."""
    return {
        "summary_focus": ["Cloud expertise"],
        "experiences_to_emphasize": ["Tech Corp"],
        "skills_to_highlight": ["Python", "AWS"],
        "achievements_to_feature": ["Led team"],
        "keywords_to_incorporate": ["microservices"],
        "sections_to_reorder": ["summary"],
    }


@pytest.fixture
def base_state() -> CVWarlockState:
    """Create a base state for testing."""
    return {
        "raw_cv": "# John Doe\n\nSenior Engineer...",
        "raw_job_spec": "# Senior Software Engineer\n\nRequirements...",
        "assume_all_tech_skills": True,
        "use_cot": True,
        "lookback_years": 4,
        "cv_data": None,
        "job_requirements": None,
        "match_analysis": None,
        "tailoring_plan": None,
        "tailored_summary": None,
        "tailored_experiences": None,
        "tailored_skills": None,
        "tailored_cv": None,
        "summary_reasoning_result": None,
        "experience_reasoning_results": None,
        "skills_reasoning_result": None,
        "generation_context": None,
        "total_refinement_iterations": 0,
        "quality_scores": None,
        "step_timings": [],
        "current_step_start": None,
        "total_generation_time": None,
        "messages": [],
        "current_step": "initialize",
        "current_step_description": "Starting",
        "errors": [],
    }


def create_mock_cot_result(
    final_content: str,  # noqa: ARG001
    quality: str = "good",
    refinement_count: int = 1,
    **extra_attrs: Any,
) -> MagicMock:
    """Create a mock CoT result with common attributes."""
    mock_critique = MagicMock()
    mock_critique.quality_level.value = quality

    mock_result = MagicMock()
    mock_result.critique = mock_critique
    mock_result.refinement_count = refinement_count

    for attr, value in extra_attrs.items():
        setattr(mock_result, attr, value)

    return mock_result


# =============================================================================
# Step Descriptions Tests
# =============================================================================

EXPECTED_WORKFLOW_STEPS = [
    "validate_inputs",
    "extract_cv",
    "extract_job",
    "analyze_match",
    "create_plan",
    "tailor_skills",
    "tailor_experiences",
    "tailor_summary",
    "assemble_cv",
]


class TestStepDescriptions:
    """Tests for step description constants."""

    @pytest.mark.parametrize(
        "descriptions,desc_type",
        [
            (STEP_DESCRIPTIONS, "CoT"),
            (STEP_DESCRIPTIONS_FAST, "Fast"),
        ],
    )
    def test_step_descriptions_has_all_steps(self, descriptions: dict, desc_type: str) -> None:
        """Test that step descriptions have all workflow steps."""
        for step in EXPECTED_WORKFLOW_STEPS:
            assert step in descriptions, f"{desc_type} missing step: {step}"

    def test_cot_descriptions_differ_from_fast(self) -> None:
        """Test that CoT descriptions are more detailed than fast."""
        assert "reasoning" in STEP_DESCRIPTIONS["tailor_skills"]
        assert "reasoning" not in STEP_DESCRIPTIONS_FAST["tailor_skills"]


# =============================================================================
# _start_step and _end_step Tests
# =============================================================================


class TestStartStep:
    """Tests for _start_step helper function."""

    def test_start_step_returns_step_info(self, base_state: CVWarlockState) -> None:
        """Test that _start_step returns correct step info."""
        result = _start_step(base_state, "extract_cv", use_cot=True)

        assert result["current_step"] == "extract_cv"
        assert "current_step_description" in result
        assert isinstance(result["current_step_start"], float)

    @pytest.mark.parametrize(
        "use_cot,should_have_reasoning",
        [(True, True), (False, False)],
    )
    def test_start_step_description_mode(
        self, base_state: CVWarlockState, use_cot: bool, should_have_reasoning: bool
    ) -> None:
        """Test that _start_step uses correct descriptions based on CoT mode."""
        result = _start_step(base_state, "tailor_skills", use_cot=use_cot)
        has_reasoning = "reasoning" in result["current_step_description"]
        assert has_reasoning == should_have_reasoning

    def test_start_step_fallback_description(self, base_state: CVWarlockState) -> None:
        """Test fallback description for unknown steps."""
        result = _start_step(base_state, "unknown_step", use_cot=True)
        assert "Running unknown_step" in result["current_step_description"]

    def test_start_step_calls_callback(self, base_state: CVWarlockState) -> None:
        """Test that _start_step calls the on_step_start callback."""
        callback = MagicMock()
        result = _start_step(base_state, "extract_cv", use_cot=True, on_step_start=callback)

        callback.assert_called_once_with("extract_cv", STEP_DESCRIPTIONS["extract_cv"])
        assert result["current_step"] == "extract_cv"

    def test_start_step_handles_callback_exception(self, base_state: CVWarlockState) -> None:
        """Test that _start_step handles callback exceptions gracefully."""
        callback = MagicMock(side_effect=Exception("Callback error"))
        result = _start_step(base_state, "extract_cv", use_cot=True, on_step_start=callback)

        callback.assert_called_once()
        assert result["current_step"] == "extract_cv"
        assert "current_step_start" in result


class TestEndStep:
    """Tests for _end_step helper function."""

    def test_end_step_records_timing(self, base_state: CVWarlockState) -> None:
        """Test that _end_step records timing information."""
        base_state["current_step_start"] = time.time() - 1.0

        result = _end_step(base_state, "extract_cv", {"some_key": "some_value"})

        assert len(result["step_timings"]) == 1
        timing = result["step_timings"][0]
        assert timing["step_name"] == "extract_cv"
        assert timing["duration_seconds"] >= 1.0
        assert "current_step_start" not in result

    def test_end_step_appends_to_existing_timings(self, base_state: CVWarlockState) -> None:
        """Test that _end_step appends to existing timings."""
        base_state["step_timings"] = [
            {
                "step_name": "validate_inputs",
                "start_time": 0,
                "end_time": 1,
                "duration_seconds": 1.0,
            }
        ]
        base_state["current_step_start"] = time.time()

        result = _end_step(base_state, "extract_cv", {})

        assert len(result["step_timings"]) == 2
        assert result["step_timings"][0]["step_name"] == "validate_inputs"
        assert result["step_timings"][1]["step_name"] == "extract_cv"

    def test_end_step_handles_missing_start_time(self, base_state: CVWarlockState) -> None:
        """Test that _end_step handles missing start time gracefully."""
        base_state["current_step_start"] = None

        result = _end_step(base_state, "extract_cv", {})

        assert result["step_timings"][0]["duration_seconds"] == 0


# =============================================================================
# create_nodes Tests
# =============================================================================


class TestCreateNodes:
    """Tests for create_nodes factory function."""

    def test_create_nodes_returns_all_nodes(self, mock_provider: MagicMock) -> None:
        """Test that create_nodes returns all required nodes."""
        nodes = create_nodes(mock_provider, use_cot=True)

        for node_name in EXPECTED_WORKFLOW_STEPS:
            assert node_name in nodes
            assert callable(nodes[node_name])

    def test_create_nodes_with_cot_disabled(self, mock_provider: MagicMock) -> None:
        """Test creating nodes with CoT disabled."""
        nodes = create_nodes(mock_provider, use_cot=False)
        # 11 nodes including extract_all and tailor_skills_and_experiences
        assert len(nodes) == 11


# =============================================================================
# Validate Inputs Node Tests
# =============================================================================


class TestValidateInputsNode:
    """Tests for the validate_inputs node."""

    def test_validate_inputs_success(
        self, mock_provider: MagicMock, base_state: CVWarlockState
    ) -> None:
        """Test validate_inputs with valid inputs."""
        nodes = create_nodes(mock_provider)
        result = nodes["validate_inputs"](base_state)

        assert result["errors"] == []
        assert "step_timings" in result
        assert result["total_refinement_iterations"] == 0

    def test_validate_inputs_initializes_step_timings(self, mock_provider: MagicMock) -> None:
        """Test that validate_inputs initializes step_timings if not present."""
        state: CVWarlockState = {
            "raw_cv": "# John Doe\n\nSenior Engineer...",
            "raw_job_spec": "# Senior Software Engineer\n\nRequirements...",
            "assume_all_tech_skills": True,
            "use_cot": True,
            "cv_data": None,
            "job_requirements": None,
            "errors": [],
        }  # type: ignore[typeddict-item]

        nodes = create_nodes(mock_provider)
        result = nodes["validate_inputs"](state)

        assert isinstance(result["step_timings"], list)

    @pytest.mark.parametrize(
        "raw_cv,raw_job_spec,expected_errors,error_contains",
        [
            ("", "# Job", 1, "CV document"),
            ("# CV", "", 1, "Job specification"),
            ("", "   ", 2, None),  # Both empty
            ("   \n\t  ", "# Job", 1, "CV"),  # Whitespace CV
        ],
    )
    def test_validate_inputs_errors(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        raw_cv: str,
        raw_job_spec: str,
        expected_errors: int,
        error_contains: str | None,
    ) -> None:
        """Test validate_inputs with various invalid inputs."""
        base_state["raw_cv"] = raw_cv
        base_state["raw_job_spec"] = raw_job_spec
        nodes = create_nodes(mock_provider)

        result = nodes["validate_inputs"](base_state)

        assert len(result["errors"]) == expected_errors
        if error_contains:
            assert any(error_contains in e for e in result["errors"])


# =============================================================================
# Extraction Node Tests
# =============================================================================


class TestExtractCVNode:
    """Tests for the extract_cv node."""

    def test_extract_cv_success(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
    ) -> None:
        """Test successful CV extraction."""
        with patch("cv_warlock.graph.nodes.CVExtractor.extract", return_value=sample_cv_data):
            nodes = create_nodes(mock_provider)
            result = nodes["extract_cv"](base_state)

        assert result["cv_data"] is sample_cv_data

    def test_extract_cv_handles_exception(
        self, mock_provider: MagicMock, base_state: CVWarlockState
    ) -> None:
        """Test that extract_cv handles exceptions gracefully."""
        mock_model = MagicMock()
        mock_provider.get_extraction_model.return_value = mock_model
        mock_model.with_structured_output.side_effect = Exception("API Error")

        nodes = create_nodes(mock_provider)
        result = nodes["extract_cv"](base_state)

        assert any("CV extraction failed" in e for e in result["errors"])


class TestExtractCVExperienceWarning:
    """Tests for extract_cv experience section validation."""

    def test_extract_cv_warns_when_experience_section_empty(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
    ) -> None:
        """Test extract_cv warns when CV has experience section but extraction returns empty."""
        # CV has experience section marker
        base_state["raw_cv"] = "# John Doe\n\n## Experience\n\nSenior Engineer at Tech Corp"

        # Create CV data with empty experiences
        empty_exp_cv = CVData(
            contact=ContactInfo(name="John Doe", email="john@example.com"),
            summary="Engineer",
            experiences=[],  # Empty experiences despite section existing
            education=[],
            skills=["Python"],
        )

        with patch("cv_warlock.graph.nodes.CVExtractor.extract", return_value=empty_exp_cv):
            nodes = create_nodes(mock_provider)
            result = nodes["extract_cv"](base_state)

        assert any("Experience section detected" in e for e in result["errors"])
        assert any("no experiences extracted" in e for e in result["errors"])

    @pytest.mark.parametrize(
        "section_marker",
        ["## Experience", "## Professional Experience", "## Work Experience"],
    )
    def test_extract_cv_warns_for_various_experience_markers(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        section_marker: str,
    ) -> None:
        """Test extract_cv warns for all supported experience section markers."""
        base_state["raw_cv"] = f"# John Doe\n\n{section_marker}\n\nSenior Engineer"

        empty_exp_cv = CVData(
            contact=ContactInfo(name="John Doe", email="john@example.com"),
            summary="Engineer",
            experiences=[],
            education=[],
            skills=["Python"],
        )

        with patch("cv_warlock.graph.nodes.CVExtractor.extract", return_value=empty_exp_cv):
            nodes = create_nodes(mock_provider)
            result = nodes["extract_cv"](base_state)

        assert any("Experience section detected" in e for e in result["errors"])


class TestExtractJobNode:
    """Tests for the extract_job node."""

    def test_extract_job_success(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test successful job extraction."""
        with patch(
            "cv_warlock.graph.nodes.JobExtractor.extract",
            return_value=sample_job_requirements,
        ):
            nodes = create_nodes(mock_provider)
            result = nodes["extract_job"](base_state)

        assert result["job_requirements"] is sample_job_requirements

    def test_extract_job_handles_exception(
        self, mock_provider: MagicMock, base_state: CVWarlockState
    ) -> None:
        """Test that extract_job handles exceptions gracefully."""
        mock_model = MagicMock()
        mock_provider.get_extraction_model.return_value = mock_model
        mock_model.with_structured_output.side_effect = Exception("Parse Error")

        nodes = create_nodes(mock_provider)
        result = nodes["extract_job"](base_state)

        assert any("Job extraction failed" in e for e in result["errors"])


class TestExtractAllNode:
    """Tests for the extract_all node (parallel CV + job extraction)."""

    def test_extract_all_success(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test successful parallel extraction."""
        with (
            patch("cv_warlock.graph.nodes.CVExtractor.extract", return_value=sample_cv_data),
            patch(
                "cv_warlock.graph.nodes.JobExtractor.extract",
                return_value=sample_job_requirements,
            ),
        ):
            nodes = create_nodes(mock_provider)
            result = nodes["extract_all"](base_state)

        assert result["cv_data"] is sample_cv_data
        assert result["job_requirements"] is sample_job_requirements

    @pytest.mark.parametrize(
        "cv_error,job_error,cv_expected,job_expected,error_count",
        [
            (Exception("CV error"), None, None, True, 1),  # CV fails
            (None, Exception("Job error"), True, None, 1),  # Job fails
            (Exception("CV"), Exception("Job"), None, None, 2),  # Both fail
        ],
    )
    def test_extract_all_failures(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        cv_error: Exception | None,
        job_error: Exception | None,
        cv_expected: bool | None,
        job_expected: bool | None,
        error_count: int,
    ) -> None:
        """Test extract_all with various failure scenarios."""
        with (
            patch(
                "cv_warlock.graph.nodes.CVExtractor.extract",
                side_effect=cv_error if cv_error else None,
                return_value=None if cv_error else sample_cv_data,
            ),
            patch(
                "cv_warlock.graph.nodes.JobExtractor.extract",
                side_effect=job_error if job_error else None,
                return_value=None if job_error else sample_job_requirements,
            ),
        ):
            nodes = create_nodes(mock_provider)
            result = nodes["extract_all"](base_state)

        if cv_expected:
            assert result["cv_data"] is sample_cv_data
        if job_expected:
            assert result["job_requirements"] is sample_job_requirements
        assert len(result["errors"]) == error_count

    def test_extract_all_warns_when_experience_section_empty(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test extract_all warns when CV has experience section but extraction returns empty."""
        # CV has experience section marker
        base_state["raw_cv"] = "# John Doe\n\n## Experience\n\nSenior Engineer at Tech Corp"

        # Create CV data with empty experiences
        empty_exp_cv = CVData(
            contact=ContactInfo(name="John Doe", email="john@example.com"),
            summary="Engineer",
            experiences=[],  # Empty experiences despite section existing
            education=[],
            skills=["Python"],
        )

        with (
            patch("cv_warlock.graph.nodes.CVExtractor.extract", return_value=empty_exp_cv),
            patch(
                "cv_warlock.graph.nodes.JobExtractor.extract",
                return_value=sample_job_requirements,
            ),
        ):
            nodes = create_nodes(mock_provider)
            result = nodes["extract_all"](base_state)

        assert result["cv_data"] is empty_exp_cv
        assert result["job_requirements"] is sample_job_requirements
        assert any("Experience section detected" in e for e in result["errors"])
        assert any("no experiences extracted" in e for e in result["errors"])


# =============================================================================
# Nodes Skip-on-Errors Tests (Parametrized)
# =============================================================================


class TestNodesSkipOnErrors:
    """Parametrized tests for nodes that skip processing when errors exist."""

    @pytest.mark.parametrize(
        "node_name,result_key",
        [
            ("analyze_match", "match_analysis"),
            ("create_plan", "tailoring_plan"),
            ("tailor_skills", "tailored_skills"),
            ("tailor_experiences", "tailored_experiences"),
            ("tailor_summary", "tailored_summary"),
            ("assemble_cv", "tailored_cv"),
            ("tailor_skills_and_experiences", "tailored_skills"),
        ],
    )
    def test_node_skips_on_errors(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        node_name: str,
        result_key: str,
    ) -> None:
        """Test that nodes skip processing when errors exist."""
        base_state["errors"] = ["Previous error"]

        nodes = create_nodes(mock_provider)
        result = nodes[node_name](base_state)

        assert result_key not in result or result.get(result_key) is None


# =============================================================================
# Analyze Match Node Tests
# =============================================================================


class TestAnalyzeMatchNode:
    """Tests for the analyze_match node."""

    @patch("cv_warlock.scoring.hybrid.HybridScorer")
    def test_analyze_match_success(
        self,
        mock_scorer_class: MagicMock,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_match_analysis: MatchAnalysis,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test successful match analysis."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["errors"] = []

        mock_scorer = MagicMock()
        mock_scorer_class.return_value = mock_scorer
        mock_scorer.score_with_plan.return_value = (sample_match_analysis, sample_tailoring_plan)

        nodes = create_nodes(mock_provider)
        result = nodes["analyze_match"](base_state)

        assert result["match_analysis"] is sample_match_analysis
        assert result["tailoring_plan"] is sample_tailoring_plan

    @patch("cv_warlock.scoring.hybrid.HybridScorer")
    def test_analyze_match_augments_skills_when_enabled(
        self,
        mock_scorer_class: MagicMock,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_match_analysis: MatchAnalysis,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test that analyze_match augments CV skills when assume_all_tech_skills is True."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["assume_all_tech_skills"] = True
        base_state["errors"] = []

        mock_scorer = MagicMock()
        mock_scorer_class.return_value = mock_scorer
        mock_scorer.score_with_plan.return_value = (sample_match_analysis, sample_tailoring_plan)

        nodes = create_nodes(mock_provider)
        result = nodes["analyze_match"](base_state)

        augmented_cv = result.get("cv_data")
        if augmented_cv:
            all_skills_lower = [s.lower() for s in augmented_cv.skills]
            assert "kubernetes" in all_skills_lower

    @patch("cv_warlock.scoring.hybrid.HybridScorer")
    def test_analyze_match_no_augmentation_when_disabled(
        self,
        mock_scorer_class: MagicMock,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_match_analysis: MatchAnalysis,
    ) -> None:
        """Test that analyze_match doesn't augment skills when disabled."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["assume_all_tech_skills"] = False
        base_state["errors"] = []
        original_skill_count = len(sample_cv_data.skills)

        mock_scorer = MagicMock()
        mock_scorer_class.return_value = mock_scorer
        mock_scorer.score.return_value = sample_match_analysis

        nodes = create_nodes(mock_provider)
        result = nodes["analyze_match"](base_state)

        if "cv_data" in result:
            assert len(result["cv_data"].skills) == original_skill_count

    @patch("cv_warlock.scoring.hybrid.HybridScorer")
    def test_analyze_match_handles_exception(
        self,
        mock_scorer_class: MagicMock,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that analyze_match handles scoring exceptions gracefully."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["errors"] = []

        mock_scorer = MagicMock()
        mock_scorer_class.return_value = mock_scorer
        mock_scorer.score.side_effect = Exception("Scoring failed")

        nodes = create_nodes(mock_provider)
        result = nodes["analyze_match"](base_state)

        assert any("Match analysis failed" in e for e in result["errors"])


# =============================================================================
# Create Plan Node Tests
# =============================================================================


class TestCreatePlanNode:
    """Tests for the create_plan node."""

    def test_create_plan_success(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_match_analysis: MatchAnalysis,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test successful tailoring plan creation."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["match_analysis"] = sample_match_analysis
        base_state["errors"] = []

        with patch(
            "cv_warlock.graph.nodes.MatchAnalyzer.create_tailoring_plan",
            return_value=sample_tailoring_plan,
        ):
            nodes = create_nodes(mock_provider)
            result = nodes["create_plan"](base_state)

        assert result["tailoring_plan"] is sample_tailoring_plan

    def test_create_plan_handles_exception(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_match_analysis: MatchAnalysis,
    ) -> None:
        """Test that create_plan handles exceptions gracefully."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["match_analysis"] = sample_match_analysis
        base_state["errors"] = []

        with patch(
            "cv_warlock.graph.nodes.MatchAnalyzer.create_tailoring_plan",
            side_effect=Exception("Plan creation failed"),
        ):
            nodes = create_nodes(mock_provider)
            result = nodes["create_plan"](base_state)

        assert any("Tailoring plan failed" in e for e in result["errors"])

    def test_create_plan_uses_precomputed_plan(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_match_analysis: MatchAnalysis,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test create_plan passes through pre-computed plan from analyze_match."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["match_analysis"] = sample_match_analysis
        base_state["tailoring_plan"] = sample_tailoring_plan  # Pre-computed by analyze_match
        base_state["errors"] = []

        with patch(
            "cv_warlock.graph.nodes.MatchAnalyzer.create_tailoring_plan",
        ) as mock_create_plan:
            nodes = create_nodes(mock_provider)
            result = nodes["create_plan"](base_state)

        # Should NOT call create_tailoring_plan since plan was pre-computed
        mock_create_plan.assert_not_called()
        # Should pass through the existing plan
        assert result["tailoring_plan"] is sample_tailoring_plan


# =============================================================================
# Tailor Skills Node Tests
# =============================================================================


class TestTailorSkillsNode:
    """Tests for the tailor_skills node."""

    def test_tailor_skills_initializes_context(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that tailor_skills initializes generation context."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements

        mock_chat_model = MagicMock()
        mock_provider.get_chat_model.return_value = mock_chat_model
        mock_result = MagicMock()
        mock_result.content = "Skills section"
        mock_chat_model.__or__ = MagicMock(return_value=mock_chat_model)
        mock_chat_model.invoke.return_value = mock_result

        nodes = create_nodes(mock_provider, use_cot=False)
        result = nodes["tailor_skills"](base_state)

        assert "generation_context" in result

    def test_tailor_skills_cot_success(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test tailor_skills with CoT enabled returns reasoning result."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["errors"] = []

        mock_reasoning = MagicMock()
        mock_reasoning.required_skills_matched = ["Python", "AWS"]
        mock_reasoning.preferred_skills_matched = ["Kubernetes"]

        mock_cot_result = create_mock_cot_result(
            final_content="**Technical Skills:** Python, AWS, Kubernetes",
            quality="good",
            refinement_count=1,
            final_skills="**Technical Skills:** Python, AWS, Kubernetes",
            reasoning=mock_reasoning,
        )

        with patch(
            "cv_warlock.graph.nodes.CVTailor.tailor_skills_with_cot",
            return_value=mock_cot_result,
        ):
            nodes = create_nodes(mock_provider, use_cot=True)
            result = nodes["tailor_skills"](base_state)

        assert result["tailored_skills"] == [mock_cot_result.final_skills]
        assert result["skills_reasoning_result"] is mock_cot_result
        assert result["total_refinement_iterations"] == 1
        assert result["quality_scores"]["skills"] == "good"

    def test_tailor_skills_handles_exception(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that tailor_skills handles exceptions gracefully."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["errors"] = []

        with patch(
            "cv_warlock.graph.nodes.CVTailor.tailor_skills_with_cot",
            side_effect=Exception("Skills tailoring error"),
        ):
            nodes = create_nodes(mock_provider, use_cot=True)
            result = nodes["tailor_skills"](base_state)

        assert any("Skills tailoring failed" in e for e in result["errors"])


# =============================================================================
# Tailor Experiences Node Tests
# =============================================================================


class TestTailorExperiencesNode:
    """Tests for the tailor_experiences node."""

    def test_tailor_experiences_direct_success(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test tailor_experiences with direct mode (no CoT)."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["tailoring_plan"] = sample_tailoring_plan
        base_state["errors"] = []

        mock_tailored = ["**Senior Engineer** at Tech Corp\n- Led team of 5"]

        with patch(
            "cv_warlock.graph.nodes.CVTailor.tailor_experiences",
            return_value=mock_tailored,
        ):
            nodes = create_nodes(mock_provider, use_cot=False)
            result = nodes["tailor_experiences"](base_state)

        assert result["tailored_experiences"] == mock_tailored

    def test_tailor_experiences_cot_success(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test tailor_experiences with CoT enabled."""
        from cv_warlock.models.reasoning import GenerationContext

        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["tailoring_plan"] = sample_tailoring_plan
        base_state["generation_context"] = GenerationContext()
        base_state["errors"] = []

        mock_exp_result = create_mock_cot_result(
            final_content="exp", quality="good", refinement_count=2
        )
        mock_context = GenerationContext(skills_demonstrated=["Python"])

        with patch(
            "cv_warlock.graph.nodes.CVTailor.tailor_experiences_with_cot",
            return_value=(["**Senior Engineer**"], [mock_exp_result], mock_context),
        ):
            nodes = create_nodes(mock_provider, use_cot=True)
            result = nodes["tailor_experiences"](base_state)

        assert result["tailored_experiences"] == ["**Senior Engineer**"]
        assert result["experience_reasoning_results"] == [mock_exp_result]
        assert result["total_refinement_iterations"] == 2
        assert result["quality_scores"]["experience_0"] == "good"

    def test_tailor_experiences_handles_exception(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test that tailor_experiences handles exceptions gracefully."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["tailoring_plan"] = sample_tailoring_plan
        base_state["errors"] = []

        with patch(
            "cv_warlock.graph.nodes.CVTailor.tailor_experiences_with_cot",
            side_effect=Exception("Experience tailoring error"),
        ):
            nodes = create_nodes(mock_provider, use_cot=True)
            result = nodes["tailor_experiences"](base_state)

        assert any("Experience tailoring failed" in e for e in result["errors"])


# =============================================================================
# Tailor Summary Node Tests
# =============================================================================


class TestTailorSummaryNode:
    """Tests for the tailor_summary node."""

    def test_tailor_summary_direct_success(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test tailor_summary with direct mode (no CoT)."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["tailoring_plan"] = sample_tailoring_plan
        base_state["tailored_skills"] = ["Python, AWS, Docker"]
        base_state["errors"] = []

        with patch(
            "cv_warlock.graph.nodes.CVTailor.tailor_summary",
            return_value="Experienced engineer with 10 years...",
        ):
            nodes = create_nodes(mock_provider, use_cot=False)
            result = nodes["tailor_summary"](base_state)

        assert result["tailored_summary"] == "Experienced engineer with 10 years..."

    def test_tailor_summary_cot_success(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test tailor_summary with CoT enabled."""
        from cv_warlock.models.reasoning import GenerationContext

        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["tailoring_plan"] = sample_tailoring_plan
        base_state["tailored_skills"] = ["Python, AWS"]
        base_state["generation_context"] = GenerationContext()
        base_state["errors"] = []

        mock_reasoning = MagicMock()
        mock_reasoning.hook_strategy = "Lead with cloud expertise"
        mock_reasoning.strongest_metric = "10 years experience"

        mock_cot_result = create_mock_cot_result(
            final_content="Seasoned cloud architect...",
            quality="excellent",
            refinement_count=0,
            final_summary="Seasoned cloud architect...",
            reasoning=mock_reasoning,
        )

        with patch(
            "cv_warlock.graph.nodes.CVTailor.tailor_summary_with_cot",
            return_value=mock_cot_result,
        ):
            nodes = create_nodes(mock_provider, use_cot=True)
            result = nodes["tailor_summary"](base_state)

        assert result["tailored_summary"] == mock_cot_result.final_summary
        assert result["summary_reasoning_result"] is mock_cot_result
        assert result["quality_scores"]["summary"] == "excellent"

    @pytest.mark.parametrize("tailored_skills", [[], None])
    def test_tailor_summary_handles_empty_skills(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
        tailored_skills: list | None,
    ) -> None:
        """Test tailor_summary handles empty/None tailored_skills."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["tailoring_plan"] = sample_tailoring_plan
        base_state["tailored_skills"] = tailored_skills
        base_state["errors"] = []

        with patch(
            "cv_warlock.graph.nodes.CVTailor.tailor_summary",
            return_value="Summary without skills preview",
        ):
            nodes = create_nodes(mock_provider, use_cot=False)
            result = nodes["tailor_summary"](base_state)

        assert result["tailored_summary"] == "Summary without skills preview"

    def test_tailor_summary_handles_exception(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test that tailor_summary handles exceptions gracefully."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["tailoring_plan"] = sample_tailoring_plan
        base_state["tailored_skills"] = ["Skills"]
        base_state["errors"] = []

        with patch(
            "cv_warlock.graph.nodes.CVTailor.tailor_summary_with_cot",
            side_effect=Exception("Summary error"),
        ):
            nodes = create_nodes(mock_provider, use_cot=True)
            result = nodes["tailor_summary"](base_state)

        assert any("Summary tailoring failed" in e for e in result["errors"])


# =============================================================================
# Assemble CV Node Tests
# =============================================================================


class TestAssembleCVNode:
    """Tests for the assemble_cv node."""

    def test_assemble_cv_computes_total_time(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
    ) -> None:
        """Test that assemble_cv computes total generation time."""
        base_state["cv_data"] = sample_cv_data
        base_state["tailored_summary"] = "Summary"
        base_state["tailored_experiences"] = ["Exp 1"]
        base_state["tailored_skills"] = ["Skills"]
        base_state["errors"] = []
        base_state["step_timings"] = [
            {"step_name": "step1", "duration_seconds": 1.0},
            {"step_name": "step2", "duration_seconds": 2.0},
        ]

        with patch("cv_warlock.graph.nodes.CVTailor.assemble_cv", return_value="# Assembled CV"):
            nodes = create_nodes(mock_provider)
            result = nodes["assemble_cv"](base_state)

        assert result["tailored_cv"] == "# Assembled CV"
        assert result["total_generation_time"] >= 3.0

    def test_assemble_cv_handles_exception(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
    ) -> None:
        """Test that assemble_cv handles exceptions gracefully."""
        base_state["cv_data"] = sample_cv_data
        base_state["tailored_summary"] = "Summary"
        base_state["tailored_experiences"] = ["Exp 1"]
        base_state["tailored_skills"] = ["Skills"]
        base_state["errors"] = []

        with patch(
            "cv_warlock.graph.nodes.CVTailor.assemble_cv",
            side_effect=Exception("Assembly error"),
        ):
            nodes = create_nodes(mock_provider)
            result = nodes["assemble_cv"](base_state)

        assert any("CV assembly failed" in e for e in result["errors"])

    def test_assemble_cv_handles_empty_skills_list(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
    ) -> None:
        """Test that assemble_cv handles empty tailored_skills list."""
        base_state["cv_data"] = sample_cv_data
        base_state["tailored_summary"] = "Summary"
        base_state["tailored_experiences"] = ["Exp 1"]
        base_state["tailored_skills"] = []
        base_state["errors"] = []

        with patch(
            "cv_warlock.graph.nodes.CVTailor.assemble_cv",
            return_value="# Final CV",
        ) as mock_assemble:
            nodes = create_nodes(mock_provider)
            result = nodes["assemble_cv"](base_state)

        mock_assemble.assert_called_once()
        call_args = mock_assemble.call_args
        assert call_args[0][3] == ""  # Fourth positional arg is skills
        assert result["tailored_cv"] == "# Final CV"

    def test_assemble_cv_adds_current_step_time(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
    ) -> None:
        """Test that assemble_cv includes current step time in total."""
        base_state["cv_data"] = sample_cv_data
        base_state["tailored_summary"] = "Summary"
        base_state["tailored_experiences"] = ["Exp 1"]
        base_state["tailored_skills"] = ["Skills"]
        base_state["errors"] = []
        base_state["step_timings"] = []
        base_state["current_step_start"] = time.time() - 0.5

        with patch("cv_warlock.graph.nodes.CVTailor.assemble_cv", return_value="# Final CV"):
            nodes = create_nodes(mock_provider)
            result = nodes["assemble_cv"](base_state)

        assert result["total_generation_time"] >= 0.5
        assert "complete" in result["current_step_description"].lower()


# =============================================================================
# Error Accumulation Tests
# =============================================================================


class TestNodeErrorAccumulation:
    """Tests for error accumulation across nodes."""

    def test_errors_accumulate_across_nodes(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
    ) -> None:
        """Test that errors accumulate and don't overwrite."""
        base_state["errors"] = ["Error 1"]

        mock_model = MagicMock()
        mock_provider.get_extraction_model.return_value = mock_model
        mock_model.with_structured_output.side_effect = Exception("Error 2")

        nodes = create_nodes(mock_provider)
        result = nodes["extract_cv"](base_state)

        assert len(result["errors"]) == 2
        assert "Error 1" in result["errors"]
        assert any("Error 2" in e or "CV extraction failed" in e for e in result["errors"])


# =============================================================================
# Tailor Skills and Experiences (Parallel) Node Tests
# =============================================================================


class TestTailorSkillsAndExperiencesNode:
    """Tests for tailor_skills_and_experiences parallel node."""

    def test_tailor_skills_and_experiences_cot_success(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test tailor_skills_and_experiences with CoT enabled."""
        from cv_warlock.models.reasoning import GenerationContext

        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["tailoring_plan"] = sample_tailoring_plan
        base_state["errors"] = []

        mock_skills_reasoning = MagicMock()
        mock_skills_reasoning.required_skills_matched = ["Python", "AWS"]
        mock_skills_reasoning.preferred_skills_matched = ["Kubernetes"]

        mock_skills_result = create_mock_cot_result(
            final_content="skills",
            quality="good",
            refinement_count=1,
            final_skills="**Technical Skills:** Python, AWS",
            reasoning=mock_skills_reasoning,
        )

        mock_exp_result = create_mock_cot_result(
            final_content="exp", quality="excellent", refinement_count=2
        )

        mock_exp_context = GenerationContext(
            skills_demonstrated=["Docker"],
            metrics_used=["led team of 5"],
            keyword_frequency={"python": 2},
        )

        with (
            patch(
                "cv_warlock.graph.nodes.CVTailor.tailor_skills_with_cot",
                return_value=mock_skills_result,
            ),
            patch(
                "cv_warlock.graph.nodes.CVTailor.tailor_experiences_with_cot",
                return_value=(["Exp 1"], [mock_exp_result], mock_exp_context),
            ),
        ):
            nodes = create_nodes(mock_provider, use_cot=True)
            result = nodes["tailor_skills_and_experiences"](base_state)

        assert result["tailored_skills"] == [mock_skills_result.final_skills]
        assert result["skills_reasoning_result"] is mock_skills_result
        assert result["quality_scores"]["skills"] == "good"
        assert result["tailored_experiences"] == ["Exp 1"]
        assert result["experience_reasoning_results"] == [mock_exp_result]
        assert result["quality_scores"]["experiences"] == ["excellent"]
        assert "Docker" in result["generation_context"].skills_demonstrated
        assert result["total_refinement_iterations"] == 3

    def test_tailor_skills_and_experiences_direct_success(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test tailor_skills_and_experiences with direct mode (no CoT)."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["tailoring_plan"] = sample_tailoring_plan
        base_state["errors"] = []

        with (
            patch(
                "cv_warlock.graph.nodes.CVTailor.tailor_skills",
                return_value="Python, AWS, Docker",
            ),
            patch(
                "cv_warlock.graph.nodes.CVTailor.tailor_experiences",
                return_value=["Exp 1", "Exp 2"],
            ),
        ):
            nodes = create_nodes(mock_provider, use_cot=False)
            result = nodes["tailor_skills_and_experiences"](base_state)

        assert result["tailored_skills"] == ["Python, AWS, Docker"]
        assert result["tailored_experiences"] == ["Exp 1", "Exp 2"]
        assert result["generation_context"] is not None

    @pytest.mark.parametrize(
        "skills_error,exp_error,skills_expected,exp_expected,error_patterns",
        [
            (
                Exception("Skills error"),
                None,
                None,
                ["Exp 1"],
                ["Skills tailoring failed"],
            ),
            (
                None,
                Exception("Exp error"),
                ["Python, AWS"],
                None,
                ["Experience tailoring failed"],
            ),
            (
                Exception("Skills error"),
                Exception("Exp error"),
                None,
                None,
                ["Skills tailoring failed", "Experience tailoring failed"],
            ),
        ],
    )
    def test_tailor_skills_and_experiences_exceptions(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
        skills_error: Exception | None,
        exp_error: Exception | None,
        skills_expected: list | None,
        exp_expected: list | None,
        error_patterns: list,
    ) -> None:
        """Test tailor_skills_and_experiences handles various exceptions."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["tailoring_plan"] = sample_tailoring_plan
        base_state["errors"] = []

        with (
            patch(
                "cv_warlock.graph.nodes.CVTailor.tailor_skills",
                side_effect=skills_error if skills_error else None,
                return_value=None if skills_error else "Python, AWS",
            ),
            patch(
                "cv_warlock.graph.nodes.CVTailor.tailor_experiences",
                side_effect=exp_error if exp_error else None,
                return_value=None if exp_error else ["Exp 1"],
            ),
        ):
            nodes = create_nodes(mock_provider, use_cot=False)
            result = nodes["tailor_skills_and_experiences"](base_state)

        if skills_expected:
            assert result["tailored_skills"] == skills_expected
        if exp_expected:
            assert result["tailored_experiences"] == exp_expected

        for pattern in error_patterns:
            assert any(pattern in e for e in result["errors"])
