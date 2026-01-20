"""End-to-end workflow tests with mocked LLM provider.

These tests validate the complete workflow pipeline without making actual LLM calls,
ensuring fast execution (<5s) while testing the full integration.
"""

from unittest.mock import MagicMock, patch

import pytest

from cv_warlock.graph.workflow import create_cv_warlock_graph, run_cv_tailoring
from cv_warlock.llm.base import LLMProvider
from cv_warlock.models.cv import ContactInfo, CVData, Education, Experience
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.models.state import MatchAnalysis, TailoringPlan


class MockLLMProvider(LLMProvider):
    """Mock LLM provider that returns predefined responses."""

    def __init__(self):
        self._provider = "mock"
        self._model = "mock-model"
        # Initialize cache attributes from parent (required for model caching)
        self._chat_model = None
        self._extraction_model = None

    def _create_chat_model(self):
        """Create a mock chat model."""
        mock = MagicMock()
        mock.invoke.return_value = MagicMock(content="Mock response")
        return mock

    def _create_extraction_model(self):
        """Create a mock extraction model."""
        mock = MagicMock()
        mock.invoke.return_value = MagicMock(content="Mock response")
        return mock

    def get_extraction_model_with_schema(self, schema):
        """Return a mock that returns appropriate Pydantic models."""
        mock = MagicMock()

        # Determine what type of extraction based on schema
        if schema == CVData:
            mock.invoke.return_value = self._mock_cv_data()
        elif schema == JobRequirements:
            mock.invoke.return_value = self._mock_job_requirements()
        elif schema == MatchAnalysis or (
            hasattr(schema, "__name__") and "Match" in schema.__name__
        ):
            mock.invoke.return_value = self._mock_match_analysis()
        elif schema == TailoringPlan or (hasattr(schema, "__name__") and "Plan" in schema.__name__):
            mock.invoke.return_value = self._mock_tailoring_plan()
        else:
            # Generic mock for other schemas
            mock.invoke.return_value = MagicMock()

        return mock

    def _mock_cv_data(self) -> CVData:
        """Create mock CV data."""
        return CVData(
            contact=ContactInfo(
                name="Test User",
                email="test@example.com",
                phone="+1234567890",
                location="Test City",
            ),
            summary="Experienced software engineer with expertise in Python and cloud technologies.",
            experiences=[
                Experience(
                    title="Senior Software Engineer",
                    company="Tech Corp",
                    start_date="January 2022",
                    end_date="Present",
                    description="Led development of cloud-native applications.",
                    achievements=["Led team of 5 engineers", "Reduced deployment time by 50%"],
                    skills_used=["Python", "AWS", "Docker"],
                ),
                Experience(
                    title="Software Engineer",
                    company="Startup Inc",
                    start_date="June 2019",
                    end_date="December 2021",
                    description="Built backend services.",
                    achievements=["Implemented microservices architecture"],
                    skills_used=["Python", "PostgreSQL"],
                ),
            ],
            education=[
                Education(
                    degree="Master of Science in Computer Science",
                    institution="Test University",
                    graduation_date="May 2019",
                )
            ],
            skills=["Python", "AWS", "Docker", "PostgreSQL", "Redis"],
            languages=["English"],
        )

    def _mock_job_requirements(self) -> JobRequirements:
        """Create mock job requirements."""
        return JobRequirements(
            job_title="Senior Software Engineer",
            company="Acme Inc",
            required_skills=["Python", "AWS", "Docker"],
            preferred_skills=["Kubernetes", "Terraform"],
            required_experience_years=5,
            required_education="Bachelor's degree",
            seniority_level="senior",
            job_type="full-time",
            remote="hybrid",
            keywords=["microservices", "cloud-native"],
            industry_terms=["SaaS"],
            soft_skills=["leadership", "communication"],
            responsibilities=["Design systems", "Lead projects"],
        )

    def _mock_match_analysis(self) -> MatchAnalysis:
        """Create mock match analysis."""
        return {
            "strong_matches": ["Python", "AWS", "Docker"],
            "partial_matches": ["Cloud experience"],
            "gaps": ["Kubernetes", "Terraform"],
            "transferable_skills": ["Redis experience"],
            "relevance_score": 0.75,
        }

    def _mock_tailoring_plan(self) -> TailoringPlan:
        """Create mock tailoring plan."""
        return {
            "summary_focus": ["Cloud-native development", "Team leadership"],
            "experiences_to_emphasize": ["Senior Software Engineer at Tech Corp"],
            "skills_to_highlight": ["Python", "AWS", "Docker"],
            "achievements_to_feature": ["Led team of 5", "Reduced deployment time"],
            "keywords_to_incorporate": ["microservices", "cloud-native"],
            "sections_to_reorder": ["summary", "experience", "skills"],
        }


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    return MockLLMProvider()


class TestE2EWorkflowWithMocks:
    """End-to-end tests using mocked LLM responses."""

    def test_workflow_graph_compiles(self, mock_llm_provider):
        """Test that the workflow graph compiles successfully."""
        with patch("cv_warlock.graph.workflow.get_llm_provider", return_value=mock_llm_provider):
            graph = create_cv_warlock_graph(provider="anthropic")
            assert graph is not None

    def test_full_workflow_with_mocks(self, sample_cv_text, sample_job_text):
        """Test the complete workflow with mocked LLM calls."""
        mock_provider = MockLLMProvider()

        # Patch the extractors, matcher, and tailor to use mock data
        with (
            patch("cv_warlock.graph.nodes.CVExtractor") as mock_cv_ext,
            patch("cv_warlock.graph.nodes.JobExtractor") as mock_job_ext,
            patch("cv_warlock.graph.nodes.MatchAnalyzer") as mock_analyzer,
            patch("cv_warlock.graph.nodes.CVTailor") as mock_tailor,
            patch("cv_warlock.scoring.hybrid.HybridScorer") as mock_scorer,
            patch("cv_warlock.graph.workflow.get_llm_provider", return_value=mock_provider),
        ):
            # Configure mock extractors
            mock_cv_ext.return_value.extract.return_value = mock_provider._mock_cv_data()
            mock_job_ext.return_value.extract.return_value = mock_provider._mock_job_requirements()

            # Configure mock scorer - uses score_with_plan() which returns (match_analysis, plan)
            mock_scorer_instance = MagicMock()
            mock_scorer_instance.score_with_plan.return_value = (
                mock_provider._mock_match_analysis(),
                mock_provider._mock_tailoring_plan(),
            )
            mock_scorer.return_value = mock_scorer_instance

            # Configure mock analyzer (fallback if plan not from scorer)
            mock_analyzer.return_value.create_tailoring_plan.return_value = (
                mock_provider._mock_tailoring_plan()
            )

            # Configure mock tailor
            mock_tailor_instance = MagicMock()
            mock_tailor_instance.tailor_summary.return_value = "Tailored summary for the role."
            mock_tailor_instance.tailor_experiences.return_value = [
                "**Senior Software Engineer** - Led cloud development"
            ]
            mock_tailor_instance.tailor_skills.return_value = "Python, AWS, Docker, Kubernetes"
            mock_tailor_instance.assemble_cv.return_value = (
                "# Test User\n\n## Summary\n\nTailored CV content"
            )
            mock_tailor.return_value = mock_tailor_instance

            # Run workflow with CoT disabled for faster execution
            result = run_cv_tailoring(
                raw_cv=sample_cv_text,
                raw_job_spec=sample_job_text,
                provider="anthropic",
                use_cot=False,  # Disable CoT for faster mock testing
            )

            # Verify workflow completed successfully
            assert result is not None
            assert result.get("errors") == [], f"Workflow errors: {result.get('errors')}"
            assert result.get("cv_data") is not None
            assert result.get("job_requirements") is not None
            assert result.get("match_analysis") is not None
            assert result.get("tailoring_plan") is not None
            assert result.get("tailored_cv") is not None

    def test_workflow_validation_catches_empty_cv(self, sample_job_text):
        """Test that validation catches empty CV input."""
        mock_provider = MockLLMProvider()

        with patch("cv_warlock.graph.workflow.get_llm_provider", return_value=mock_provider):
            result = run_cv_tailoring(
                raw_cv="",  # Empty CV
                raw_job_spec=sample_job_text,
                provider="anthropic",
            )

            assert len(result.get("errors", [])) > 0
            assert any("empty" in err.lower() for err in result["errors"])

    def test_workflow_validation_catches_empty_job(self, sample_cv_text):
        """Test that validation catches empty job spec input."""
        mock_provider = MockLLMProvider()

        with patch("cv_warlock.graph.workflow.get_llm_provider", return_value=mock_provider):
            result = run_cv_tailoring(
                raw_cv=sample_cv_text,
                raw_job_spec="",  # Empty job spec
                provider="anthropic",
            )

            assert len(result.get("errors", [])) > 0
            assert any("empty" in err.lower() for err in result["errors"])

    def test_workflow_tracks_step_timings(self, sample_cv_text, sample_job_text):
        """Test that workflow tracks timing for each step."""
        mock_provider = MockLLMProvider()

        with (
            patch("cv_warlock.graph.nodes.CVExtractor") as mock_cv_ext,
            patch("cv_warlock.graph.nodes.JobExtractor") as mock_job_ext,
            patch("cv_warlock.graph.nodes.MatchAnalyzer") as mock_analyzer,
            patch("cv_warlock.graph.nodes.CVTailor") as mock_tailor,
            patch("cv_warlock.graph.nodes.HybridScorer", create=True) as mock_scorer,
            patch("cv_warlock.graph.workflow.get_llm_provider", return_value=mock_provider),
        ):
            # Configure mocks
            mock_cv_ext.return_value.extract.return_value = mock_provider._mock_cv_data()
            mock_job_ext.return_value.extract.return_value = mock_provider._mock_job_requirements()
            mock_scorer.return_value.score.return_value = mock_provider._mock_match_analysis()
            mock_analyzer.return_value.create_tailoring_plan.return_value = (
                mock_provider._mock_tailoring_plan()
            )
            mock_tailor.return_value.tailor_summary.return_value = "Summary"
            mock_tailor.return_value.tailor_experiences.return_value = ["Experience"]
            mock_tailor.return_value.tailor_skills.return_value = "Skills"
            mock_tailor.return_value.assemble_cv.return_value = "# CV"

            result = run_cv_tailoring(
                raw_cv=sample_cv_text,
                raw_job_spec=sample_job_text,
                provider="anthropic",
                use_cot=False,
            )

            # Check that timings were recorded
            step_timings = result.get("step_timings", [])
            assert len(step_timings) > 0, "No step timings recorded"

            # Verify expected steps are tracked
            step_names = [t.get("step_name") or t.step_name for t in step_timings]
            assert "validate_inputs" in step_names
            assert "extract_all" in step_names  # Parallel extraction

    def test_progress_callback_receives_updates(self, sample_cv_text, sample_job_text):
        """Test that progress callback is invoked for each step."""
        mock_provider = MockLLMProvider()
        progress_updates = []

        def progress_callback(step_name, description, elapsed):
            progress_updates.append((step_name, description, elapsed))

        with (
            patch("cv_warlock.graph.nodes.CVExtractor") as mock_cv_ext,
            patch("cv_warlock.graph.nodes.JobExtractor") as mock_job_ext,
            patch("cv_warlock.graph.nodes.MatchAnalyzer") as mock_analyzer,
            patch("cv_warlock.graph.nodes.CVTailor") as mock_tailor,
            patch("cv_warlock.graph.nodes.HybridScorer", create=True) as mock_scorer,
            patch("cv_warlock.graph.workflow.get_llm_provider", return_value=mock_provider),
        ):
            # Configure mocks
            mock_cv_ext.return_value.extract.return_value = mock_provider._mock_cv_data()
            mock_job_ext.return_value.extract.return_value = mock_provider._mock_job_requirements()
            mock_scorer.return_value.score.return_value = mock_provider._mock_match_analysis()
            mock_analyzer.return_value.create_tailoring_plan.return_value = (
                mock_provider._mock_tailoring_plan()
            )
            mock_tailor.return_value.tailor_summary.return_value = "Summary"
            mock_tailor.return_value.tailor_experiences.return_value = ["Experience"]
            mock_tailor.return_value.tailor_skills.return_value = "Skills"
            mock_tailor.return_value.assemble_cv.return_value = "# CV"

            run_cv_tailoring(
                raw_cv=sample_cv_text,
                raw_job_spec=sample_job_text,
                provider="anthropic",
                use_cot=False,
                progress_callback=progress_callback,
            )

            # Verify progress updates were received
            assert len(progress_updates) > 0, "No progress updates received"

            # Check that key steps were reported
            reported_steps = [step for step, _, _ in progress_updates]
            assert "validate_inputs" in reported_steps
            assert "extract_all" in reported_steps  # Parallel extraction


class TestE2EWorkflowSmoke:
    """Smoke tests for workflow structure without any LLM calls."""

    def test_initial_state_structure(self):
        """Test that initial state has all required fields."""
        from cv_warlock.models.state import CVWarlockState

        # Verify CVWarlockState has expected keys
        expected_keys = [
            "raw_cv",
            "raw_job_spec",
            "cv_data",
            "job_requirements",
            "match_analysis",
            "tailoring_plan",
            "tailored_cv",
            "errors",
        ]

        # Check type hints
        annotations = CVWarlockState.__annotations__
        for key in expected_keys:
            assert key in annotations, f"Missing key in CVWarlockState: {key}"

    def test_workflow_nodes_are_created(self):
        """Test that all expected nodes are created."""
        from cv_warlock.graph.nodes import create_nodes

        mock_provider = MockLLMProvider()
        nodes = create_nodes(mock_provider, use_cot=False)

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
            assert node_name in nodes, f"Missing node: {node_name}"
            assert callable(nodes[node_name]), f"Node {node_name} is not callable"

    def test_validate_inputs_node_works(self, sample_cv_text, sample_job_text):
        """Test validate_inputs node in isolation."""
        from cv_warlock.graph.nodes import create_nodes
        from cv_warlock.models.state import CVWarlockState

        mock_provider = MockLLMProvider()
        nodes = create_nodes(mock_provider, use_cot=False)

        # Create minimal state
        state: CVWarlockState = {
            "raw_cv": sample_cv_text,
            "raw_job_spec": sample_job_text,
            "assume_all_tech_skills": True,
            "use_cot": False,
            "lookback_years": None,
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
            "current_step": "start",
            "current_step_description": "",
            "errors": [],
        }

        # Run validation node
        result = nodes["validate_inputs"](state)

        # Should pass with no errors
        assert result.get("errors") == []
        assert result.get("current_step") == "validate_inputs"
