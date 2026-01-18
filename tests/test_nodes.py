"""Tests for LangGraph workflow nodes."""

import time
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


class TestStepDescriptions:
    """Tests for step description constants."""

    def test_step_descriptions_cot_has_all_steps(self) -> None:
        """Test that STEP_DESCRIPTIONS has all workflow steps."""
        expected_steps = [
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

        for step in expected_steps:
            assert step in STEP_DESCRIPTIONS

    def test_step_descriptions_fast_has_all_steps(self) -> None:
        """Test that STEP_DESCRIPTIONS_FAST has all workflow steps."""
        expected_steps = [
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

        for step in expected_steps:
            assert step in STEP_DESCRIPTIONS_FAST

    def test_cot_descriptions_differ_from_fast(self) -> None:
        """Test that CoT descriptions are more detailed than fast."""
        # CoT descriptions mention reasoning
        assert "reasoning" in STEP_DESCRIPTIONS["tailor_skills"]
        # Fast descriptions are simpler
        assert "reasoning" not in STEP_DESCRIPTIONS_FAST["tailor_skills"]


class TestStartStep:
    """Tests for _start_step helper function."""

    def test_start_step_returns_step_info(self, base_state: CVWarlockState) -> None:
        """Test that _start_step returns correct step info."""
        result = _start_step(base_state, "extract_cv", use_cot=True)

        assert result["current_step"] == "extract_cv"
        assert "current_step_description" in result
        assert "current_step_start" in result
        assert isinstance(result["current_step_start"], float)

    def test_start_step_uses_cot_descriptions(self, base_state: CVWarlockState) -> None:
        """Test that _start_step uses CoT descriptions when enabled."""
        result = _start_step(base_state, "tailor_skills", use_cot=True)

        assert "reasoning" in result["current_step_description"]

    def test_start_step_uses_fast_descriptions(self, base_state: CVWarlockState) -> None:
        """Test that _start_step uses fast descriptions when CoT disabled."""
        result = _start_step(base_state, "tailor_skills", use_cot=False)

        assert "reasoning" not in result["current_step_description"]

    def test_start_step_fallback_description(self, base_state: CVWarlockState) -> None:
        """Test fallback description for unknown steps."""
        result = _start_step(base_state, "unknown_step", use_cot=True)

        assert "Running unknown_step" in result["current_step_description"]


class TestEndStep:
    """Tests for _end_step helper function."""

    def test_end_step_records_timing(self, base_state: CVWarlockState) -> None:
        """Test that _end_step records timing information."""
        base_state["current_step_start"] = time.time() - 1.0  # 1 second ago

        updates = {"some_key": "some_value"}
        result = _end_step(base_state, "extract_cv", updates)

        assert "step_timings" in result
        assert len(result["step_timings"]) == 1
        timing = result["step_timings"][0]
        assert timing["step_name"] == "extract_cv"
        assert timing["duration_seconds"] >= 1.0
        assert result["current_step_start"] is None

    def test_end_step_appends_to_existing_timings(self, base_state: CVWarlockState) -> None:
        """Test that _end_step appends to existing timings."""
        existing_timing = {
            "step_name": "validate_inputs",
            "start_time": 0,
            "end_time": 1,
            "duration_seconds": 1.0,
        }
        base_state["step_timings"] = [existing_timing]
        base_state["current_step_start"] = time.time()

        updates = {}
        result = _end_step(base_state, "extract_cv", updates)

        assert len(result["step_timings"]) == 2
        assert result["step_timings"][0]["step_name"] == "validate_inputs"
        assert result["step_timings"][1]["step_name"] == "extract_cv"

    def test_end_step_handles_missing_start_time(self, base_state: CVWarlockState) -> None:
        """Test that _end_step handles missing start time gracefully."""
        base_state["current_step_start"] = None

        updates = {}
        result = _end_step(base_state, "extract_cv", updates)

        # Should still create timing entry
        assert "step_timings" in result
        timing = result["step_timings"][0]
        assert timing["duration_seconds"] == 0


class TestCreateNodes:
    """Tests for create_nodes factory function."""

    def test_create_nodes_returns_all_nodes(self, mock_provider: MagicMock) -> None:
        """Test that create_nodes returns all required nodes."""
        nodes = create_nodes(mock_provider, use_cot=True)

        expected_nodes = [
            "validate_inputs",
            "extract_cv",
            "extract_job",
            "analyze_match",
            "create_plan",
            "tailor_summary",
            "tailor_experiences",
            "tailor_skills",
            "assemble_cv",
        ]

        for node_name in expected_nodes:
            assert node_name in nodes
            assert callable(nodes[node_name])

    def test_create_nodes_with_cot_disabled(self, mock_provider: MagicMock) -> None:
        """Test creating nodes with CoT disabled."""
        nodes = create_nodes(mock_provider, use_cot=False)

        # Should still have all nodes
        assert len(nodes) == 9


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

    def test_validate_inputs_empty_cv(
        self, mock_provider: MagicMock, base_state: CVWarlockState
    ) -> None:
        """Test validate_inputs with empty CV."""
        base_state["raw_cv"] = ""
        nodes = create_nodes(mock_provider)

        result = nodes["validate_inputs"](base_state)

        assert len(result["errors"]) == 1
        assert "CV document" in result["errors"][0]

    def test_validate_inputs_empty_job_spec(
        self, mock_provider: MagicMock, base_state: CVWarlockState
    ) -> None:
        """Test validate_inputs with empty job spec."""
        base_state["raw_job_spec"] = ""
        nodes = create_nodes(mock_provider)

        result = nodes["validate_inputs"](base_state)

        assert len(result["errors"]) == 1
        assert "Job specification" in result["errors"][0]

    def test_validate_inputs_both_empty(
        self, mock_provider: MagicMock, base_state: CVWarlockState
    ) -> None:
        """Test validate_inputs with both inputs empty."""
        base_state["raw_cv"] = ""
        base_state["raw_job_spec"] = "   "  # Whitespace only
        nodes = create_nodes(mock_provider)

        result = nodes["validate_inputs"](base_state)

        assert len(result["errors"]) == 2

    def test_validate_inputs_whitespace_cv(
        self, mock_provider: MagicMock, base_state: CVWarlockState
    ) -> None:
        """Test that whitespace-only CV is treated as empty."""
        base_state["raw_cv"] = "   \n\t  "
        nodes = create_nodes(mock_provider)

        result = nodes["validate_inputs"](base_state)

        assert len(result["errors"]) >= 1
        assert any("CV" in e for e in result["errors"])


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
        assert "errors" not in result or not result.get("errors")

    def test_extract_cv_handles_exception(
        self, mock_provider: MagicMock, base_state: CVWarlockState
    ) -> None:
        """Test that extract_cv handles exceptions gracefully."""
        mock_model = MagicMock()
        mock_provider.get_extraction_model.return_value = mock_model
        mock_model.with_structured_output.side_effect = Exception("API Error")

        nodes = create_nodes(mock_provider)
        result = nodes["extract_cv"](base_state)

        assert "errors" in result
        assert any("CV extraction failed" in e for e in result["errors"])


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
    ) -> None:
        """Test successful match analysis."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["errors"] = []  # Ensure no errors

        mock_scorer = MagicMock()
        mock_scorer_class.return_value = mock_scorer
        mock_scorer.score.return_value = sample_match_analysis

        nodes = create_nodes(mock_provider)
        result = nodes["analyze_match"](base_state)

        assert result["match_analysis"] is sample_match_analysis

    @patch("cv_warlock.scoring.hybrid.HybridScorer")
    def test_analyze_match_augments_skills_when_enabled(
        self,
        mock_scorer_class: MagicMock,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_match_analysis: MatchAnalysis,
    ) -> None:
        """Test that analyze_match augments CV skills when assume_all_tech_skills is True."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["assume_all_tech_skills"] = True
        base_state["errors"] = []  # Ensure no errors

        mock_scorer = MagicMock()
        mock_scorer_class.return_value = mock_scorer
        mock_scorer.score.return_value = sample_match_analysis

        nodes = create_nodes(mock_provider)
        result = nodes["analyze_match"](base_state)

        # The augmented CV should have extra skills
        augmented_cv = result.get("cv_data")
        if augmented_cv:
            # Should include job required/preferred skills
            all_skills_lower = [s.lower() for s in augmented_cv.skills]
            assert "kubernetes" in all_skills_lower  # from preferred_skills

    def test_analyze_match_skips_on_errors(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
    ) -> None:
        """Test that analyze_match skips processing if errors exist."""
        base_state["errors"] = ["Previous error"]

        nodes = create_nodes(mock_provider)
        result = nodes["analyze_match"](base_state)

        # Should not add match_analysis
        assert "match_analysis" not in result or result.get("match_analysis") is None


class TestCreatePlanNode:
    """Tests for the create_plan node."""

    def test_create_plan_skips_on_errors(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
    ) -> None:
        """Test that create_plan skips processing if errors exist."""
        base_state["errors"] = ["Previous error"]

        nodes = create_nodes(mock_provider)
        result = nodes["create_plan"](base_state)

        assert "tailoring_plan" not in result or result.get("tailoring_plan") is None


class TestTailorSkillsNode:
    """Tests for the tailor_skills node."""

    def test_tailor_skills_skips_on_errors(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
    ) -> None:
        """Test that tailor_skills skips processing if errors exist."""
        base_state["errors"] = ["Previous error"]

        nodes = create_nodes(mock_provider)
        result = nodes["tailor_skills"](base_state)

        assert "tailored_skills" not in result or result.get("tailored_skills") is None

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


class TestTailorExperiencesNode:
    """Tests for the tailor_experiences node."""

    def test_tailor_experiences_skips_on_errors(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
    ) -> None:
        """Test that tailor_experiences skips processing if errors exist."""
        base_state["errors"] = ["Previous error"]

        nodes = create_nodes(mock_provider)
        result = nodes["tailor_experiences"](base_state)

        assert "tailored_experiences" not in result or result.get("tailored_experiences") is None


class TestTailorSummaryNode:
    """Tests for the tailor_summary node."""

    def test_tailor_summary_skips_on_errors(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
    ) -> None:
        """Test that tailor_summary skips processing if errors exist."""
        base_state["errors"] = ["Previous error"]

        nodes = create_nodes(mock_provider)
        result = nodes["tailor_summary"](base_state)

        assert "tailored_summary" not in result or result.get("tailored_summary") is None


class TestAssembleCVNode:
    """Tests for the assemble_cv node."""

    def test_assemble_cv_skips_on_errors(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
    ) -> None:
        """Test that assemble_cv skips processing if errors exist."""
        base_state["errors"] = ["Previous error"]

        nodes = create_nodes(mock_provider)
        result = nodes["assemble_cv"](base_state)

        assert "tailored_cv" not in result or result.get("tailored_cv") is None

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
        base_state["errors"] = []  # Ensure no errors
        base_state["step_timings"] = [
            {"step_name": "step1", "duration_seconds": 1.0},
            {"step_name": "step2", "duration_seconds": 2.0},
        ]

        with patch("cv_warlock.graph.nodes.CVTailor.assemble_cv", return_value="# Assembled CV"):
            nodes = create_nodes(mock_provider)
            result = nodes["assemble_cv"](base_state)

        assert result["tailored_cv"] == "# Assembled CV"
        assert result["total_generation_time"] >= 3.0  # At least previous steps


class TestNodeErrorAccumulation:
    """Tests for error accumulation across nodes."""

    def test_errors_accumulate_across_nodes(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
    ) -> None:
        """Test that errors accumulate and don't overwrite."""
        base_state["errors"] = ["Error 1"]

        # Make extraction fail
        mock_model = MagicMock()
        mock_provider.get_extraction_model.return_value = mock_model
        mock_model.with_structured_output.side_effect = Exception("Error 2")

        nodes = create_nodes(mock_provider)
        result = nodes["extract_cv"](base_state)

        # Should have both errors
        assert len(result["errors"]) == 2
        assert "Error 1" in result["errors"]
        assert any("Error 2" in e or "CV extraction failed" in e for e in result["errors"])


class TestCreatePlanNodeSuccess:
    """Tests for create_plan node success and error paths."""

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


class TestTailorSkillsNodeCoT:
    """Tests for tailor_skills node with CoT enabled."""

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

        # Create mock CoT result
        mock_reasoning = MagicMock()
        mock_reasoning.required_skills_matched = ["Python", "AWS"]
        mock_reasoning.preferred_skills_matched = ["Kubernetes"]

        mock_critique = MagicMock()
        mock_critique.quality_level.value = "good"

        mock_cot_result = MagicMock()
        mock_cot_result.final_skills = "**Technical Skills:** Python, AWS, Kubernetes"
        mock_cot_result.reasoning = mock_reasoning
        mock_cot_result.critique = mock_critique
        mock_cot_result.refinement_count = 1

        with patch(
            "cv_warlock.graph.nodes.CVTailor.tailor_skills_with_cot",
            return_value=mock_cot_result,
        ):
            nodes = create_nodes(mock_provider, use_cot=True)
            result = nodes["tailor_skills"](base_state)

        assert result["tailored_skills"] == [mock_cot_result.final_skills]
        assert result["skills_reasoning_result"] is mock_cot_result
        assert result["generation_context"] is not None
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


class TestTailorExperiencesNodePaths:
    """Tests for tailor_experiences node success and error paths."""

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

        mock_tailored_texts = ["**Senior Engineer** at Tech Corp\n- Led team"]

        # Create mock experience result
        mock_critique = MagicMock()
        mock_critique.quality_level.value = "good"

        mock_exp_result = MagicMock()
        mock_exp_result.refinement_count = 2
        mock_exp_result.critique = mock_critique

        mock_context = GenerationContext(skills_demonstrated=["Python"])

        with patch(
            "cv_warlock.graph.nodes.CVTailor.tailor_experiences_with_cot",
            return_value=(mock_tailored_texts, [mock_exp_result], mock_context),
        ):
            nodes = create_nodes(mock_provider, use_cot=True)
            result = nodes["tailor_experiences"](base_state)

        assert result["tailored_experiences"] == mock_tailored_texts
        assert result["experience_reasoning_results"] == [mock_exp_result]
        assert result["generation_context"] is mock_context
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


class TestTailorSummaryNodePaths:
    """Tests for tailor_summary node success and error paths."""

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

        mock_summary = "Experienced engineer with 10 years..."

        with patch(
            "cv_warlock.graph.nodes.CVTailor.tailor_summary",
            return_value=mock_summary,
        ):
            nodes = create_nodes(mock_provider, use_cot=False)
            result = nodes["tailor_summary"](base_state)

        assert result["tailored_summary"] == mock_summary

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

        # Create mock CoT result
        mock_reasoning = MagicMock()
        mock_reasoning.hook_strategy = "Lead with cloud expertise"
        mock_reasoning.strongest_metric = "10 years experience"

        mock_critique = MagicMock()
        mock_critique.quality_level.value = "excellent"

        mock_cot_result = MagicMock()
        mock_cot_result.final_summary = "Seasoned cloud architect..."
        mock_cot_result.reasoning = mock_reasoning
        mock_cot_result.critique = mock_critique
        mock_cot_result.refinement_count = 0

        with patch(
            "cv_warlock.graph.nodes.CVTailor.tailor_summary_with_cot",
            return_value=mock_cot_result,
        ):
            nodes = create_nodes(mock_provider, use_cot=True)
            result = nodes["tailor_summary"](base_state)

        assert result["tailored_summary"] == mock_cot_result.final_summary
        assert result["summary_reasoning_result"] is mock_cot_result
        assert result["generation_context"].established_identity == "Lead with cloud expertise"
        assert result["quality_scores"]["summary"] == "excellent"

    def test_tailor_summary_handles_empty_skills(
        self,
        mock_provider: MagicMock,
        base_state: CVWarlockState,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test tailor_summary handles empty tailored_skills list."""
        base_state["cv_data"] = sample_cv_data
        base_state["job_requirements"] = sample_job_requirements
        base_state["tailoring_plan"] = sample_tailoring_plan
        base_state["tailored_skills"] = []  # Empty list
        base_state["errors"] = []

        mock_summary = "Summary without skills preview"

        with patch(
            "cv_warlock.graph.nodes.CVTailor.tailor_summary",
            return_value=mock_summary,
        ):
            nodes = create_nodes(mock_provider, use_cot=False)
            result = nodes["tailor_summary"](base_state)

        assert result["tailored_summary"] == mock_summary

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


class TestAssembleCVNodePaths:
    """Tests for assemble_cv node additional paths."""

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
        base_state["tailored_skills"] = []  # Empty list
        base_state["errors"] = []

        with patch(
            "cv_warlock.graph.nodes.CVTailor.assemble_cv",
            return_value="# Final CV",
        ) as mock_assemble:
            nodes = create_nodes(mock_provider)
            result = nodes["assemble_cv"](base_state)

        # Should pass empty string for skills
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
        base_state["current_step_start"] = time.time() - 0.5  # Started 0.5s ago

        with patch(
            "cv_warlock.graph.nodes.CVTailor.assemble_cv",
            return_value="# Final CV",
        ):
            nodes = create_nodes(mock_provider)
            result = nodes["assemble_cv"](base_state)

        # Total time should include at least the current step duration
        assert result["total_generation_time"] >= 0.5
        assert "complete" in result["current_step_description"].lower()


class TestAnalyzeMatchNodeEdgeCases:
    """Tests for analyze_match node edge cases."""

    @patch("cv_warlock.scoring.hybrid.HybridScorer")
    def test_analyze_match_no_skill_augmentation_when_disabled(
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
        base_state["assume_all_tech_skills"] = False  # Disabled
        base_state["errors"] = []
        original_skill_count = len(sample_cv_data.skills)

        mock_scorer = MagicMock()
        mock_scorer_class.return_value = mock_scorer
        mock_scorer.score.return_value = sample_match_analysis

        nodes = create_nodes(mock_provider)
        result = nodes["analyze_match"](base_state)

        # CV data should not be modified (or not returned)
        if "cv_data" in result:
            # If returned, skills should be unchanged
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
