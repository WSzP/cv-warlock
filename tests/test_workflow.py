"""Tests for LangGraph workflow assembly and execution."""

from unittest.mock import MagicMock, patch

import pytest

from cv_warlock.graph.workflow import create_cv_warlock_graph, run_cv_tailoring
from cv_warlock.models.cv import ContactInfo, CVData, Education, Experience
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.models.state import MatchAnalysis, TailoringPlan


@pytest.fixture
def sample_cv_text() -> str:
    """Sample CV text for testing."""
    return """# John Doe

## Summary
Experienced software engineer with 10+ years building scalable systems.

## Experience

### Senior Software Engineer | Tech Corp
*January 2020 - Present*
- Led team of 5 engineers
- Reduced deployment time by 50%

## Skills
Python, AWS, Docker, Kubernetes
"""


@pytest.fixture
def sample_job_text() -> str:
    """Sample job posting text for testing."""
    return """# Senior Software Engineer

## About the Role
We are looking for an experienced engineer to join our team.

## Requirements
- 5+ years of experience with Python
- Experience with AWS and Docker
- Kubernetes experience preferred

## Responsibilities
- Design and implement scalable systems
- Lead technical projects
"""


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


class TestCreateCVWarlockGraph:
    """Tests for the create_cv_warlock_graph function."""

    @patch("cv_warlock.graph.workflow.get_settings")
    @patch("cv_warlock.graph.workflow.get_llm_provider")
    def test_create_graph_returns_compiled_graph(
        self,
        mock_get_provider: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """Test that create_cv_warlock_graph returns a compiled graph."""
        mock_settings = MagicMock()
        mock_settings.provider = "anthropic"
        mock_settings.model = "claude-sonnet-4-5-20250929"
        mock_settings.anthropic_api_key = "test-key"
        mock_get_settings.return_value = mock_settings

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        graph = create_cv_warlock_graph()

        # Graph should be compiled (have invoke method)
        assert hasattr(graph, "invoke")
        assert callable(graph.invoke)

    @patch("cv_warlock.graph.workflow.get_settings")
    @patch("cv_warlock.graph.workflow.get_llm_provider")
    def test_create_graph_with_explicit_provider(
        self,
        mock_get_provider: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """Test creating graph with explicit provider parameter."""
        mock_settings = MagicMock()
        mock_settings.openai_api_key = "test-openai-key"
        mock_get_settings.return_value = mock_settings

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        create_cv_warlock_graph(provider="openai", model="gpt-5.2")

        mock_get_provider.assert_called_once_with("openai", "gpt-5.2", "test-openai-key")

    @patch("cv_warlock.graph.workflow.get_settings")
    @patch("cv_warlock.graph.workflow.get_llm_provider")
    def test_create_graph_with_explicit_api_key(
        self,
        mock_get_provider: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """Test creating graph with explicit API key."""
        mock_settings = MagicMock()
        mock_settings.provider = "anthropic"
        mock_settings.model = "claude-sonnet-4-5-20250929"
        mock_get_settings.return_value = mock_settings

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        create_cv_warlock_graph(api_key="explicit-key")

        mock_get_provider.assert_called_once_with(
            "anthropic", "claude-sonnet-4-5-20250929", "explicit-key"
        )

    @patch("cv_warlock.graph.workflow.get_settings")
    @patch("cv_warlock.graph.workflow.get_llm_provider")
    def test_create_graph_with_cot_enabled(
        self,
        mock_get_provider: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """Test creating graph with CoT enabled."""
        mock_settings = MagicMock()
        mock_settings.provider = "anthropic"
        mock_settings.model = "claude-sonnet-4-5-20250929"
        mock_settings.anthropic_api_key = "key"
        mock_get_settings.return_value = mock_settings

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        graph = create_cv_warlock_graph(use_cot=True)

        assert graph is not None

    @patch("cv_warlock.graph.workflow.get_settings")
    @patch("cv_warlock.graph.workflow.get_llm_provider")
    def test_create_graph_with_cot_disabled(
        self,
        mock_get_provider: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """Test creating graph with CoT disabled."""
        mock_settings = MagicMock()
        mock_settings.provider = "anthropic"
        mock_settings.model = "claude-sonnet-4-5-20250929"
        mock_settings.anthropic_api_key = "key"
        mock_get_settings.return_value = mock_settings

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        graph = create_cv_warlock_graph(use_cot=False)

        assert graph is not None

    @patch("cv_warlock.graph.workflow.get_settings")
    @patch("cv_warlock.graph.workflow.get_llm_provider")
    def test_create_graph_google_provider(
        self,
        mock_get_provider: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """Test creating graph with Google provider."""
        mock_settings = MagicMock()
        mock_settings.google_api_key = "google-key"
        mock_get_settings.return_value = mock_settings

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        create_cv_warlock_graph(provider="google", model="gemini-3-flash-preview")

        mock_get_provider.assert_called_once_with(
            "google", "gemini-3-flash-preview", "google-key"
        )


class TestRunCVTailoring:
    """Tests for the run_cv_tailoring function."""

    @patch("cv_warlock.graph.workflow.create_cv_warlock_graph")
    def test_run_cv_tailoring_creates_initial_state(
        self,
        mock_create_graph: MagicMock,
        sample_cv_text: str,
        sample_job_text: str,
    ) -> None:
        """Test that run_cv_tailoring creates proper initial state."""
        mock_graph = MagicMock()
        mock_create_graph.return_value = mock_graph

        mock_graph.invoke.return_value = {
            "tailored_cv": "# Tailored CV",
            "errors": [],
        }

        run_cv_tailoring(sample_cv_text, sample_job_text)

        # Verify invoke was called with correct initial state
        mock_graph.invoke.assert_called_once()
        initial_state = mock_graph.invoke.call_args[0][0]

        assert initial_state["raw_cv"] == sample_cv_text
        assert initial_state["raw_job_spec"] == sample_job_text
        assert initial_state["assume_all_tech_skills"] is True
        assert initial_state["use_cot"] is True
        assert initial_state["errors"] == []

    @patch("cv_warlock.graph.workflow.create_cv_warlock_graph")
    def test_run_cv_tailoring_returns_final_state(
        self,
        mock_create_graph: MagicMock,
        sample_cv_text: str,
        sample_job_text: str,
    ) -> None:
        """Test that run_cv_tailoring returns the final workflow state."""
        mock_graph = MagicMock()
        mock_create_graph.return_value = mock_graph

        expected_result = {
            "tailored_cv": "# John Doe\n\n## Summary\n...",
            "errors": [],
            "total_generation_time": 10.5,
        }
        mock_graph.invoke.return_value = expected_result

        result = run_cv_tailoring(sample_cv_text, sample_job_text)

        assert result["tailored_cv"] == expected_result["tailored_cv"]
        assert "total_generation_time" in result

    @patch("cv_warlock.graph.workflow.create_cv_warlock_graph")
    def test_run_cv_tailoring_with_assume_all_tech_skills_false(
        self,
        mock_create_graph: MagicMock,
        sample_cv_text: str,
        sample_job_text: str,
    ) -> None:
        """Test running with assume_all_tech_skills disabled."""
        mock_graph = MagicMock()
        mock_create_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {"tailored_cv": "CV", "errors": []}

        run_cv_tailoring(
            sample_cv_text,
            sample_job_text,
            assume_all_tech_skills=False,
        )

        initial_state = mock_graph.invoke.call_args[0][0]
        assert initial_state["assume_all_tech_skills"] is False

    @patch("cv_warlock.graph.workflow.create_cv_warlock_graph")
    def test_run_cv_tailoring_with_cot_disabled(
        self,
        mock_create_graph: MagicMock,
        sample_cv_text: str,
        sample_job_text: str,
    ) -> None:
        """Test running with CoT disabled."""
        mock_graph = MagicMock()
        mock_create_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {"tailored_cv": "CV", "errors": []}

        run_cv_tailoring(sample_cv_text, sample_job_text, use_cot=False)

        mock_create_graph.assert_called_once()
        call_kwargs = mock_create_graph.call_args[1]
        assert call_kwargs.get("use_cot") is False

    @patch("cv_warlock.graph.workflow.create_cv_warlock_graph")
    def test_run_cv_tailoring_with_custom_lookback_years(
        self,
        mock_create_graph: MagicMock,
        sample_cv_text: str,
        sample_job_text: str,
    ) -> None:
        """Test running with custom lookback_years."""
        mock_graph = MagicMock()
        mock_create_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {"tailored_cv": "CV", "errors": []}

        run_cv_tailoring(sample_cv_text, sample_job_text, lookback_years=10)

        initial_state = mock_graph.invoke.call_args[0][0]
        assert initial_state["lookback_years"] == 10

    @patch("cv_warlock.graph.workflow.create_cv_warlock_graph")
    def test_run_cv_tailoring_with_progress_callback(
        self,
        mock_create_graph: MagicMock,
        sample_cv_text: str,
        sample_job_text: str,
    ) -> None:
        """Test running with progress callback."""
        mock_graph = MagicMock()
        mock_create_graph.return_value = mock_graph

        # Simulate streaming events
        mock_graph.stream.return_value = [
            {"validate_inputs": {"current_step": "validate_inputs", "errors": []}},
            {"extract_cv": {"current_step": "extract_cv", "cv_data": None}},
        ]

        callback_calls = []

        def progress_callback(step: str, desc: str, elapsed: float) -> None:
            callback_calls.append((step, desc, elapsed))

        run_cv_tailoring(
            sample_cv_text,
            sample_job_text,
            progress_callback=progress_callback,
        )

        # Should have called stream instead of invoke
        mock_graph.stream.assert_called_once()
        # Should have received progress callbacks
        assert len(callback_calls) >= 1

    @patch("cv_warlock.graph.workflow.create_cv_warlock_graph")
    def test_run_cv_tailoring_with_provider_override(
        self,
        mock_create_graph: MagicMock,
        sample_cv_text: str,
        sample_job_text: str,
    ) -> None:
        """Test running with provider override."""
        mock_graph = MagicMock()
        mock_create_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {"tailored_cv": "CV", "errors": []}

        run_cv_tailoring(
            sample_cv_text,
            sample_job_text,
            provider="openai",
            model="gpt-5.2",
            api_key="test-key",
        )

        mock_create_graph.assert_called_once_with(
            "openai", "gpt-5.2", "test-key", use_cot=True
        )

    @patch("cv_warlock.graph.workflow.create_cv_warlock_graph")
    def test_run_cv_tailoring_adds_total_generation_time(
        self,
        mock_create_graph: MagicMock,
        sample_cv_text: str,
        sample_job_text: str,
    ) -> None:
        """Test that total_generation_time is added to result."""
        mock_graph = MagicMock()
        mock_create_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {"tailored_cv": "CV", "errors": []}

        result = run_cv_tailoring(sample_cv_text, sample_job_text)

        assert "total_generation_time" in result
        assert isinstance(result["total_generation_time"], float)
        assert result["total_generation_time"] >= 0


class TestWorkflowStepDescriptions:
    """Tests for workflow step descriptions."""

    @patch("cv_warlock.graph.workflow.create_cv_warlock_graph")
    def test_cot_step_descriptions(
        self,
        mock_create_graph: MagicMock,
        sample_cv_text: str,
        sample_job_text: str,
    ) -> None:
        """Test that CoT mode uses detailed step descriptions."""
        mock_graph = MagicMock()
        mock_create_graph.return_value = mock_graph

        # Create fake streaming events
        mock_graph.stream.return_value = [
            {
                "tailor_skills": {
                    "current_step": "tailor_skills",
                    "current_step_description": "Adding job skills to CV (reasoning → generating)...",
                }
            },
        ]

        callback_calls = []

        def callback(step: str, desc: str, elapsed: float) -> None:
            callback_calls.append((step, desc))

        run_cv_tailoring(
            sample_cv_text,
            sample_job_text,
            progress_callback=callback,
            use_cot=True,
        )

        # Should have a description with reasoning
        skill_calls = [c for c in callback_calls if c[0] == "tailor_skills"]
        if skill_calls:
            assert "reasoning" in skill_calls[0][1]

    @patch("cv_warlock.graph.workflow.create_cv_warlock_graph")
    def test_fast_step_descriptions(
        self,
        mock_create_graph: MagicMock,
        sample_cv_text: str,
        sample_job_text: str,
    ) -> None:
        """Test that fast mode uses simpler step descriptions."""
        mock_graph = MagicMock()
        mock_create_graph.return_value = mock_graph

        mock_graph.stream.return_value = [
            {
                "tailor_skills": {
                    "current_step": "tailor_skills",
                    "current_step_description": "Adding job skills to CV...",
                }
            },
        ]

        callback_calls = []

        def callback(step: str, desc: str, elapsed: float) -> None:
            callback_calls.append((step, desc))

        run_cv_tailoring(
            sample_cv_text,
            sample_job_text,
            progress_callback=callback,
            use_cot=False,
        )

        mock_create_graph.assert_called_with(None, None, None, use_cot=False)


class TestWorkflowInitialState:
    """Tests for workflow initial state setup."""

    @patch("cv_warlock.graph.workflow.create_cv_warlock_graph")
    def test_initial_state_has_all_required_fields(
        self,
        mock_create_graph: MagicMock,
        sample_cv_text: str,
        sample_job_text: str,
    ) -> None:
        """Test that initial state has all required fields."""
        mock_graph = MagicMock()
        mock_create_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {}

        run_cv_tailoring(sample_cv_text, sample_job_text)

        initial_state = mock_graph.invoke.call_args[0][0]

        required_fields = [
            "raw_cv",
            "raw_job_spec",
            "assume_all_tech_skills",
            "use_cot",
            "lookback_years",
            "cv_data",
            "job_requirements",
            "match_analysis",
            "tailoring_plan",
            "tailored_summary",
            "tailored_experiences",
            "tailored_skills",
            "tailored_cv",
            "summary_reasoning_result",
            "experience_reasoning_results",
            "skills_reasoning_result",
            "generation_context",
            "total_refinement_iterations",
            "quality_scores",
            "step_timings",
            "current_step_start",
            "total_generation_time",
            "messages",
            "current_step",
            "current_step_description",
            "errors",
        ]

        for field in required_fields:
            assert field in initial_state, f"Missing field: {field}"

    @patch("cv_warlock.graph.workflow.create_cv_warlock_graph")
    def test_initial_state_has_correct_default_values(
        self,
        mock_create_graph: MagicMock,
        sample_cv_text: str,
        sample_job_text: str,
    ) -> None:
        """Test that initial state has correct default values."""
        mock_graph = MagicMock()
        mock_create_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {}

        run_cv_tailoring(sample_cv_text, sample_job_text)

        initial_state = mock_graph.invoke.call_args[0][0]

        # Check defaults
        assert initial_state["cv_data"] is None
        assert initial_state["job_requirements"] is None
        assert initial_state["match_analysis"] is None
        assert initial_state["tailoring_plan"] is None
        assert initial_state["tailored_summary"] is None
        assert initial_state["tailored_experiences"] is None
        assert initial_state["tailored_skills"] is None
        assert initial_state["tailored_cv"] is None
        assert initial_state["total_refinement_iterations"] == 0
        assert initial_state["quality_scores"] is None
        assert initial_state["step_timings"] == []
        assert initial_state["errors"] == []
        assert initial_state["current_step"] == "start"
