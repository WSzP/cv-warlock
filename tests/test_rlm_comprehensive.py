"""Comprehensive tests for RLM module to improve coverage.

Tests for:
- orchestrator.py: parsing, helpers, final answer processing
- rlm_nodes.py: all node functions and helpers
- chunking.py: edge cases
- environment.py: sandbox modes
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cv_warlock.models.cv import ContactInfo, CVData, Experience
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.rlm.chunking import CVChunker, JobChunker
from cv_warlock.rlm.environment import REPLEnvironment
from cv_warlock.rlm.models import (
    ActionType,
    CVChunks,
    JobChunks,
    ModelAction,
    RLMConfig,
    RLMResult,
    TrajectoryStep,
)
from cv_warlock.rlm.orchestrator import RLMOrchestrator

# =============================================================================
# Fixtures - Using REAL sample data from examples/
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "FINAL(Test answer)"
    mock_model.invoke.return_value = mock_response
    provider.get_chat_model.return_value = mock_model
    return provider


@pytest.fixture
def sample_cv_text():
    """Load REAL sample CV from examples/sample_cv.md."""
    cv_path = PROJECT_ROOT / "examples" / "sample_cv.md"
    return cv_path.read_text(encoding="utf-8")


@pytest.fixture
def sample_job_text():
    """Load REAL sample job posting from examples/sample_job_posting.md."""
    job_path = PROJECT_ROOT / "examples" / "sample_job_posting.md"
    return job_path.read_text(encoding="utf-8")


@pytest.fixture
def sample_state(sample_cv_text, sample_job_text):
    """Sample workflow state."""
    return {
        "raw_cv": sample_cv_text,
        "raw_job_spec": sample_job_text,
        "use_rlm": True,
        "errors": [],
    }


@pytest.fixture
def large_state(sample_cv_text, sample_job_text):
    """Large workflow state to trigger RLM threshold."""
    return {
        "raw_cv": sample_cv_text * 10,
        "raw_job_spec": sample_job_text * 10,
        "use_rlm": True,
        "errors": [],
    }


def create_mock_standard_nodes(**overrides):
    """Create a complete mock standard nodes dict with optional overrides."""
    base_nodes = {
        "validate_inputs": MagicMock(return_value={}),
        "extract_cv": MagicMock(return_value={"cv_data": None, "current_step": "extract_cv"}),
        "extract_job": MagicMock(
            return_value={"job_requirements": None, "current_step": "extract_job"}
        ),
        "extract_all": MagicMock(
            return_value={
                "cv_data": None,
                "job_requirements": None,
                "current_step": "extract_all",
            }
        ),
        "analyze_match": MagicMock(
            return_value={"match_analysis": None, "current_step": "analyze_match"}
        ),
        "create_plan": MagicMock(return_value={}),
        "tailor_skills": MagicMock(return_value={}),
        "tailor_experiences": MagicMock(return_value={}),
        "tailor_skills_and_experiences": MagicMock(return_value={}),
        "tailor_summary": MagicMock(return_value={}),
        "assemble_cv": MagicMock(return_value={}),
    }
    for node_name, mock_return in overrides.items():
        if isinstance(mock_return, dict):
            base_nodes[node_name] = MagicMock(return_value=mock_return)
        else:
            base_nodes[node_name] = mock_return
    return base_nodes


def create_valid_cv_data(name: str = "Test", skills: list | None = None) -> CVData:
    """Create a valid CVData for testing."""
    return CVData(
        contact=ContactInfo(name=name, email=f"{name.lower()}@test.com"),
        summary=f"{name} summary",
        experiences=[],
        education=[],
        skills=skills or ["Python"],
    )


def create_valid_job_requirements(title: str = "Developer") -> JobRequirements:
    """Create valid JobRequirements for testing."""
    return JobRequirements(
        job_title=title,
        required_skills=["Python"],
        preferred_skills=[],
        responsibilities=[],
    )


def create_match_analysis_result(score: float = 0.8) -> dict[str, Any]:
    """Create a match analysis result dict."""
    return {
        "match_analysis": {
            "strong_matches": ["Python"],
            "partial_matches": [],
            "gaps": [],
            "transferable_skills": [],
            "relevance_score": score,
        },
        "current_step": "analyze_match",
    }


def create_timeout_step() -> TrajectoryStep:
    """Create a timeout trajectory step."""
    return TrajectoryStep(
        step_number=1,
        action_type=ActionType.FINAL,
        model_output="TIMEOUT",
        parsed_action=None,
        execution_result="RLM timeout after 30s",
        duration_ms=0,
    )


def create_rlm_result(
    answer: Any = None,
    success: bool = True,
    trajectory: list | None = None,
    sub_call_count: int = 0,
    total_iterations: int = 1,
    error: str | None = None,
) -> RLMResult:
    """Create an RLMResult for testing."""
    return RLMResult(
        answer=answer,
        trajectory=trajectory or [],
        sub_call_count=sub_call_count,
        total_iterations=total_iterations,
        success=success,
        error=error,
    )


# =============================================================================
# Orchestrator Tests
# =============================================================================


class TestRLMOrchestratorParsing:
    """Tests for orchestrator parsing methods."""

    @pytest.mark.parametrize(
        "output,expected_type,content_check",
        [
            ("After analysis, FINAL(The candidate is great)", ActionType.FINAL, "great"),
            ("I've stored it. FINAL_VAR(analysis_result)", ActionType.FINAL, "analysis_result"),
            ("Let me think...", ActionType.CODE, "provide Python code"),
        ],
    )
    def test_parse_patterns(self, mock_llm_provider, output, expected_type, content_check):
        """Test parsing various output patterns."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        action = orchestrator._parse_model_output(output)
        assert action.action_type == expected_type
        assert content_check in action.content

    def test_parse_code_block(self, mock_llm_provider):
        """Test parsing code block."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        output = """Let me analyze.

```python
skills = cv_text.split(',')
print(len(skills))
```

This counts skills."""
        action = orchestrator._parse_model_output(output)
        assert action.action_type == ActionType.CODE
        assert "cv_text.split" in action.content

    def test_parse_code_with_query(self, mock_llm_provider):
        """Test parsing code containing rlm_query."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        output = """```python
result = rlm_query(cv_text, "What are the main skills?")
print(result)
```"""
        action = orchestrator._parse_model_output(output)
        assert action.action_type == ActionType.QUERY
        assert action.context_var == "cv_text"
        assert "skills" in action.question.lower()

    def test_parse_multiple_code_blocks_combined(self, mock_llm_provider):
        """Test that multiple code blocks are combined."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        output = """```python
step1 = "First"
print(step1)
```

```python
step2 = step1 + " and Second"
print(step2)
```"""
        action = orchestrator._parse_model_output(output)
        assert action.action_type == ActionType.CODE
        assert "step1" in action.content
        assert "step2" in action.content

    def test_final_inside_code_block_not_matched(self, mock_llm_provider):
        """Test that FINAL inside code block is not matched as action."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        output = """```python
# This is just a comment: FINAL(ignore this)
result = "actual computation"
print(result)
```"""
        action = orchestrator._parse_model_output(output)
        assert action.action_type == ActionType.CODE
        assert "actual computation" in action.content


class TestRLMOrchestratorHelpers:
    """Tests for orchestrator helper methods."""

    def test_register_helpers(self, mock_llm_provider, sample_cv_text, sample_job_text):
        """Test that helper functions are registered correctly."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        orchestrator._register_helpers(env)

        assert "rlm_query" in env._functions
        assert "find_keyword" in env._functions
        assert "find_sections" in env._functions

    @pytest.mark.parametrize(
        "code,expected_in_output",
        [
            ('positions = find_keyword("python", cv_text)\nprint(len(positions))', None),
            ("sections = find_sections(cv_text)\nprint(list(sections.keys()))", None),
        ],
    )
    def test_helper_execution(
        self, mock_llm_provider, sample_cv_text, sample_job_text, code, expected_in_output
    ):
        """Test helper function execution."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        orchestrator._register_helpers(env)
        result = env.execute(code)
        assert result.success

    def test_find_sections_no_headers(self, mock_llm_provider):
        """Test find_sections with text that has no headers."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text="Plain text", job_text="Plain job")
        orchestrator._register_helpers(env)
        result = env.execute('sections = find_sections("No headers here")\nprint(sections)')
        assert result.success
        assert "content" in result.output


class TestRLMOrchestratorFinalAnswer:
    """Tests for final answer processing."""

    def test_process_final_answer_string(self, mock_llm_provider, sample_cv_text, sample_job_text):
        """Test processing a string final answer."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        result = orchestrator._process_final_answer("The answer is 42", None, env)
        assert result == "The answer is 42"

    def test_process_final_answer_variable(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test processing a variable reference final answer."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        env.set_variable("my_result", "Variable content")
        result = orchestrator._process_final_answer("my_result", None, env)
        assert result == "Variable content"

    def test_process_final_answer_json_schema(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test processing JSON with schema."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        json_content = '{"max_iterations": 10, "max_sub_calls": 5}'
        result = orchestrator._process_final_answer(json_content, RLMConfig, env)
        assert isinstance(result, RLMConfig)
        assert result.max_iterations == 10

    def test_process_final_answer_invalid_json(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test processing invalid JSON with schema triggers LLM extraction."""
        mock_structured_model = MagicMock()
        mock_extracted_config = RLMConfig(max_iterations=5)
        mock_structured_model.invoke.return_value = mock_extracted_config
        mock_llm_provider.get_extraction_model.return_value.with_structured_output.return_value = (
            mock_structured_model
        )

        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        result = orchestrator._process_final_answer("not valid json", RLMConfig, env)
        assert isinstance(result, RLMConfig)
        assert result.max_iterations == 5


class TestRLMOrchestratorFallback:
    """Tests for fallback answer extraction."""

    @pytest.mark.parametrize(
        "var_name,var_value,schema,expected_type",
        [
            ("result", "Fallback result", None, str),
            ("answer", "Fallback answer", None, str),
            ("result", {"max_iterations": 15}, RLMConfig, RLMConfig),
            ("result", '{"max_iterations": 25}', RLMConfig, RLMConfig),
        ],
    )
    def test_extract_fallback_variants(
        self,
        mock_llm_provider,
        sample_cv_text,
        sample_job_text,
        var_name,
        var_value,
        schema,
        expected_type,
    ):
        """Test extracting fallback from different variable types."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        env.variables[var_name] = var_value
        answer = orchestrator._extract_fallback_answer(env, schema)
        assert isinstance(answer, expected_type)

    def test_extract_fallback_all_variables(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extracting all variables as fallback."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        env.variables["custom_var"] = "custom value"
        answer = orchestrator._extract_fallback_answer(env, None)
        assert isinstance(answer, dict)
        assert "custom_var" in answer

    def test_extract_fallback_empty(self, mock_llm_provider, sample_cv_text, sample_job_text):
        """Test extracting fallback with no variables returns None."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        answer = orchestrator._extract_fallback_answer(env, None)
        assert answer is None


class TestRLMOrchestratorComplete:
    """Tests for the complete orchestration loop."""

    def test_complete_with_immediate_final(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test complete with model giving immediate FINAL answer."""
        orchestrator = RLMOrchestrator(mock_llm_provider, config=RLMConfig(max_iterations=5))
        result = orchestrator.complete(
            task="Analyze the CV",
            cv_text=sample_cv_text,
            job_text=sample_job_text,
        )
        assert result.success
        assert result.answer == "Test answer"
        assert result.total_iterations == 1

    def test_complete_with_code_execution(self, sample_cv_text, sample_job_text):
        """Test complete with code execution."""
        provider = MagicMock()
        mock_model = MagicMock()
        responses = [
            MagicMock(content="```python\nskills = len(cv_text)\nprint(skills)\n```"),
            MagicMock(content="FINAL(Analysis complete)"),
        ]
        mock_model.invoke.side_effect = responses
        provider.get_chat_model.return_value = mock_model

        orchestrator = RLMOrchestrator(provider, config=RLMConfig(max_iterations=5))
        result = orchestrator.complete(
            task="Analyze the CV",
            cv_text=sample_cv_text,
            job_text=sample_job_text,
        )
        assert result.success
        assert result.total_iterations == 2

    @pytest.mark.parametrize("max_iterations", [2, 3])
    def test_complete_max_iterations(self, sample_cv_text, sample_job_text, max_iterations):
        """Test complete hitting max iterations."""
        provider = MagicMock()
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="```python\nprint('iteration')\n```")
        provider.get_chat_model.return_value = mock_model

        config = RLMConfig(max_iterations=max_iterations, timeout_seconds=300)
        orchestrator = RLMOrchestrator(provider, config=config)
        result = orchestrator.complete(
            task="Analyze",
            cv_text=sample_cv_text,
            job_text=sample_job_text,
        )
        assert result.total_iterations == max_iterations

    def test_complete_with_exception(self, sample_cv_text, sample_job_text):
        """Test complete handling exceptions."""
        provider = MagicMock()
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("API Error")
        provider.get_chat_model.return_value = mock_model

        orchestrator = RLMOrchestrator(provider)
        result = orchestrator.complete(
            task="Analyze",
            cv_text=sample_cv_text,
            job_text=sample_job_text,
        )
        assert not result.success
        assert "API Error" in result.error

    def test_complete_with_sub_call(self, sample_cv_text, sample_job_text):
        """Test complete with sub-model query."""
        root_provider = MagicMock()
        sub_provider = MagicMock()
        root_model = MagicMock()
        sub_model = MagicMock()

        root_responses = [
            MagicMock(
                content='```python\nresult = rlm_query(cv_text, "What skills?")\nprint(result)\n```'
            ),
            MagicMock(content="FINAL(Done)"),
        ]
        root_model.invoke.side_effect = root_responses
        root_provider.get_chat_model.return_value = root_model
        sub_model.invoke.return_value = MagicMock(content="Python, JavaScript")
        sub_provider.get_chat_model.return_value = sub_model

        orchestrator = RLMOrchestrator(
            root_provider, sub_provider, config=RLMConfig(max_iterations=5)
        )
        result = orchestrator.complete(
            task="Analyze",
            cv_text=sample_cv_text,
            job_text=sample_job_text,
        )
        assert result.success
        assert result.sub_call_count == 1


class TestRLMOrchestratorSubCall:
    """Tests for sub-call execution."""

    def test_execute_sub_call_basic(self, sample_cv_text, sample_job_text):
        """Test basic sub-call execution."""
        root_provider = MagicMock()
        sub_provider = MagicMock()
        sub_model = MagicMock()
        sub_model.invoke.return_value = MagicMock(content="Sub-model answer")
        sub_provider.get_chat_model.return_value = sub_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        result = orchestrator._execute_sub_call("Some context", "What is this?", env)

        assert result.answer == "Sub-model answer"
        assert orchestrator.sub_call_count == 1

    def test_execute_sub_call_variable_resolution(self, sample_cv_text, sample_job_text):
        """Test sub-call with variable name resolution."""
        root_provider = MagicMock()
        sub_provider = MagicMock()
        sub_model = MagicMock()
        sub_model.invoke.return_value = MagicMock(content="Answer")
        sub_provider.get_chat_model.return_value = sub_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        env.set_variable("my_var", "Variable content to query")
        result = orchestrator._execute_sub_call("my_var", "What about this?", env)

        assert result.answer == "Answer"

    def test_execute_sub_call_truncation(self, sample_cv_text, sample_job_text):
        """Test sub-call with long context truncation."""
        root_provider = MagicMock()
        sub_provider = MagicMock()
        sub_model = MagicMock()
        sub_model.invoke.return_value = MagicMock(content="Answer")
        sub_provider.get_chat_model.return_value = sub_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        long_context = "A" * 10000
        result = orchestrator._execute_sub_call(long_context, "Question?", env)

        assert result.answer == "Answer"
        assert "truncated" in result.context_preview or len(result.context_preview) <= 203

    def test_execute_sub_call_error(self, sample_cv_text, sample_job_text):
        """Test sub-call with error."""
        root_provider = MagicMock()
        sub_provider = MagicMock()
        sub_model = MagicMock()
        sub_model.invoke.side_effect = Exception("Sub-model error")
        sub_provider.get_chat_model.return_value = sub_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        result = orchestrator._execute_sub_call("context", "question?", env)

        assert "Error" in result.answer


class TestCreateRLMOrchestrator:
    """Tests for factory function."""

    @pytest.mark.parametrize(
        "root_provider,root_model,sub_provider,sub_model",
        [
            (None, None, None, None),  # defaults
            ("google", "gemini-3-pro-preview", None, None),
            ("openai", "gpt-5.2", None, None),
            ("anthropic", "claude-opus-4-5-20251101", "openai", "gpt-5-mini"),
            ("anthropic", "claude-opus-4-5-20251101", None, "claude-sonnet-4-5-20250929"),
        ],
    )
    def test_create_orchestrator_variants(self, root_provider, root_model, sub_provider, sub_model):
        """Test factory with various provider configurations."""
        with patch("cv_warlock.llm.base.get_llm_provider") as mock_get:
            mock_provider = MagicMock()
            mock_get.return_value = mock_provider

            from cv_warlock.rlm.orchestrator import create_rlm_orchestrator

            kwargs = {}
            if root_provider:
                kwargs["root_provider"] = root_provider
            if root_model:
                kwargs["root_model"] = root_model
            if sub_provider:
                kwargs["sub_provider"] = sub_provider
            if sub_model:
                kwargs["sub_model"] = sub_model

            orchestrator = create_rlm_orchestrator(**kwargs)

            assert orchestrator.root_provider == mock_provider
            assert mock_get.call_count == 2


# =============================================================================
# RLM Nodes Tests
# =============================================================================


class TestRLMNodesHelpers:
    """Tests for RLM node helper functions."""

    def test_convert_trajectory(self):
        """Test trajectory conversion."""
        from cv_warlock.graph.rlm_nodes import _convert_trajectory

        trajectory = [
            TrajectoryStep(
                step_number=1,
                action_type=ActionType.CODE,
                model_output="test",
                parsed_action=ModelAction(action_type=ActionType.CODE, content="print(1)"),
                execution_result="1",
                sub_call_made=False,
                duration_ms=100.0,
            ),
            TrajectoryStep(
                step_number=2,
                action_type=ActionType.FINAL,
                model_output="done",
                parsed_action=ModelAction(action_type=ActionType.FINAL, content="result"),
                execution_result="Final",
                sub_call_made=True,
                duration_ms=50.0,
            ),
        ]

        rlm_result = create_rlm_result(answer="test", trajectory=trajectory, sub_call_count=1)
        converted = _convert_trajectory(rlm_result)

        assert len(converted) == 2
        assert converted[0]["step_number"] == 1
        assert converted[0]["action_type"] == "code"
        assert converted[1]["sub_call_made"] is True

    @pytest.mark.parametrize(
        "enabled,used,rlm_result,expected_iterations,expected_sub_calls",
        [
            (True, False, None, 0, 0),
            (
                True,
                True,
                create_rlm_result(
                    answer="test",
                    sub_call_count=3,
                    total_iterations=5,
                ),
                5,
                3,
            ),
        ],
    )
    def test_create_rlm_metadata(
        self, enabled, used, rlm_result, expected_iterations, expected_sub_calls
    ):
        """Test metadata creation with various inputs."""
        from cv_warlock.graph.rlm_nodes import _create_rlm_metadata

        metadata = _create_rlm_metadata(enabled=enabled, used=used, rlm_result=rlm_result)

        assert metadata["enabled"] is enabled
        assert metadata["used"] is used
        assert metadata["total_iterations"] == expected_iterations
        assert metadata["sub_call_count"] == expected_sub_calls


class TestRLMNodesCreation:
    """Tests for RLM node creation and execution."""

    def test_create_rlm_nodes(self, mock_llm_provider):
        """Test creating RLM nodes."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        nodes = create_rlm_nodes(mock_llm_provider)
        expected_nodes = [
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
        for node in expected_nodes:
            assert node in nodes


class TestRLMNodesExtraction:
    """Parameterized tests for RLM extraction nodes."""

    def test_extract_cv_rlm_disabled(self, mock_llm_provider, sample_state):
        """Test extract_cv when RLM is disabled - includes metadata."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        state = {**sample_state, "use_rlm": False}
        expected_result = create_valid_cv_data("Test")

        with patch("cv_warlock.graph.nodes.create_nodes") as mock_create:
            mock_create.return_value = create_mock_standard_nodes(
                extract_cv={"cv_data": expected_result, "current_step": "extract_cv"}
            )
            nodes = create_rlm_nodes(mock_llm_provider)
            result = nodes["extract_cv"](state)

            assert result["cv_data"] == expected_result
            assert result["rlm_metadata"]["enabled"] is False

    def test_extract_job_rlm_disabled(self, mock_llm_provider, sample_state):
        """Test extract_job when RLM is disabled - returns standard result."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        state = {**sample_state, "use_rlm": False}
        expected_result = create_valid_job_requirements("Developer")

        with patch("cv_warlock.graph.nodes.create_nodes") as mock_create:
            mock_create.return_value = create_mock_standard_nodes(
                extract_job={"job_requirements": expected_result, "current_step": "extract_job"}
            )
            nodes = create_rlm_nodes(mock_llm_provider)
            result = nodes["extract_job"](state)

            assert result["job_requirements"] == expected_result
            # extract_job doesn't add rlm_metadata when disabled

    def test_extract_cv_rlm_below_threshold(self, mock_llm_provider, sample_state):
        """Test extract_cv when below size threshold - includes metadata."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        expected_result = create_valid_cv_data("Test")

        with patch("cv_warlock.graph.nodes.create_nodes") as mock_create:
            mock_create.return_value = create_mock_standard_nodes(
                extract_cv={"cv_data": expected_result, "current_step": "extract_cv"}
            )
            config = RLMConfig(size_threshold=100000)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_cv"](sample_state)

            assert result["cv_data"] == expected_result
            assert result["rlm_metadata"]["enabled"] is True
            assert result["rlm_metadata"]["used"] is False

    def test_extract_job_rlm_below_threshold(self, mock_llm_provider, sample_state):
        """Test extract_job when below size threshold - returns standard result."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        expected_result = create_valid_job_requirements("Developer")

        with patch("cv_warlock.graph.nodes.create_nodes") as mock_create:
            mock_create.return_value = create_mock_standard_nodes(
                extract_job={"job_requirements": expected_result, "current_step": "extract_job"}
            )
            config = RLMConfig(size_threshold=100000)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_job"](sample_state)

            assert result["job_requirements"] == expected_result
            # extract_job doesn't add rlm_metadata when below threshold

    @pytest.mark.parametrize(
        "node_name,result_key,create_result",
        [
            ("extract_cv", "cv_data", lambda: create_valid_cv_data("Test")),
            ("extract_job", "job_requirements", lambda: create_valid_job_requirements()),
        ],
    )
    def test_extraction_rlm_success(
        self, mock_llm_provider, large_state, node_name, result_key, create_result
    ):
        """Test extraction with successful RLM result."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        expected_result = create_result()

        with patch.object(RLMOrchestrator, "complete") as mock_complete:
            mock_complete.return_value = create_rlm_result(
                answer=expected_result, sub_call_count=1, total_iterations=2
            )
            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes[node_name](large_state)

            assert result[result_key] == expected_result
            assert result["rlm_metadata"]["used"] is True

    def test_extract_cv_wrong_type_fallback(self, mock_llm_provider, large_state):
        """Test extract_cv falls back when RLM returns wrong type."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        fallback_result = create_valid_cv_data("Fallback")

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create,
        ):
            mock_complete.return_value = create_rlm_result(answer="Not a CVData")
            mock_create.return_value = create_mock_standard_nodes(
                extract_cv={"cv_data": fallback_result, "current_step": "extract_cv"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_cv"](large_state)

            assert result["cv_data"] == fallback_result
            assert result["rlm_metadata"]["used"] is False

    def test_extract_job_wrong_type_fallback(self, mock_llm_provider, large_state):
        """Test extract_job falls back when RLM returns wrong type."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        fallback_result = create_valid_job_requirements("Fallback")

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create,
        ):
            mock_complete.return_value = create_rlm_result(answer="Not a JobRequirements")
            mock_create.return_value = create_mock_standard_nodes(
                extract_job={"job_requirements": fallback_result, "current_step": "extract_job"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_job"](large_state)

            assert result["job_requirements"] == fallback_result

    @pytest.mark.parametrize(
        "node_name,result_key,create_fallback,create_standard_result,error_pattern",
        [
            (
                "extract_cv",
                "cv_data",
                lambda: create_valid_cv_data("Fallback"),
                lambda cv: {"cv_data": cv, "current_step": "extract_cv"},
                "RLM extraction fallback",
            ),
            (
                "extract_job",
                "job_requirements",
                lambda: create_valid_job_requirements("Fallback"),
                lambda job: {"job_requirements": job, "current_step": "extract_job"},
                "RLM job extraction fallback",
            ),
        ],
    )
    def test_extraction_exception_fallback(
        self,
        mock_llm_provider,
        large_state,
        node_name,
        result_key,
        create_fallback,
        create_standard_result,
        error_pattern,
    ):
        """Test extraction falls back when exception occurs."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        fallback_result = create_fallback()

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create,
        ):
            mock_complete.side_effect = Exception("RLM crashed!")
            mock_create.return_value = create_mock_standard_nodes(
                **{node_name: create_standard_result(fallback_result)}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes[node_name](large_state)

            assert result[result_key] == fallback_result
            assert any(error_pattern in e for e in result["errors"])


class TestAnalyzeMatchRLM:
    """Tests for analyze_match RLM node."""

    def test_analyze_match_rlm_dict_answer(self, mock_llm_provider, large_state):
        """Test analyze_match_rlm with dict answer."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        with patch.object(RLMOrchestrator, "complete") as mock_complete:
            mock_complete.return_value = create_rlm_result(
                answer={
                    "strong_matches": ["Python"],
                    "partial_matches": ["Cloud"],
                    "gaps": ["Kubernetes"],
                    "transferable_skills": ["Leadership"],
                    "relevance_score": 0.75,
                },
                sub_call_count=1,
                total_iterations=2,
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["analyze_match"](large_state)

            assert result["match_analysis"]["relevance_score"] == 0.75

    def test_analyze_match_rlm_disabled(self, mock_llm_provider, sample_state):
        """Test analyze_match_rlm when RLM is disabled."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        state = {**sample_state, "use_rlm": False}

        with patch("cv_warlock.graph.nodes.create_nodes") as mock_create:
            mock_create.return_value = create_mock_standard_nodes(
                analyze_match=create_match_analysis_result(0.8)
            )
            nodes = create_rlm_nodes(mock_llm_provider)
            result = nodes["analyze_match"](state)

            assert result["match_analysis"]["relevance_score"] == 0.8

    @pytest.mark.parametrize(
        "rlm_answer,expected_score",
        [
            ("Not a dict answer", 0.6),  # Wrong type
            (None, 0.5),  # Failure
        ],
    )
    def test_analyze_match_fallback(
        self, mock_llm_provider, large_state, rlm_answer, expected_score
    ):
        """Test analyze_match_rlm fallback scenarios."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create,
        ):
            success = rlm_answer is not None
            mock_complete.return_value = create_rlm_result(
                answer=rlm_answer,
                success=success,
                error=None if success else "RLM failed",
            )
            mock_create.return_value = create_mock_standard_nodes(
                analyze_match=create_match_analysis_result(expected_score)
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["analyze_match"](large_state)

            assert result["match_analysis"]["relevance_score"] == expected_score


class TestRLMNodesOnStepStartCallback:
    """Tests for on_step_start callback handling in RLM nodes."""

    @pytest.mark.parametrize(
        "node_name,create_result",
        [
            ("extract_cv", lambda: create_valid_cv_data("Test")),
            ("extract_job", lambda: create_valid_job_requirements()),
            (
                "analyze_match",
                lambda: {
                    "strong_matches": ["Python"],
                    "partial_matches": [],
                    "gaps": [],
                    "transferable_skills": [],
                    "relevance_score": 0.8,
                },
            ),
        ],
    )
    def test_node_calls_on_step_start(
        self, mock_llm_provider, large_state, node_name, create_result
    ):
        """Test that nodes call on_step_start callback."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        callback = MagicMock()

        with patch.object(RLMOrchestrator, "complete") as mock_complete:
            mock_complete.return_value = create_rlm_result(
                answer=create_result(), sub_call_count=1, total_iterations=2
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config, on_step_start=callback)
            nodes[node_name](large_state)

            callback.assert_called_once()
            call_args = callback.call_args
            assert call_args[0][0] == node_name
            assert "RLM" in call_args[0][1]

    @pytest.mark.parametrize(
        "node_name,create_result,result_key",
        [
            ("extract_cv", lambda: create_valid_cv_data("Test"), "cv_data"),
            ("extract_job", lambda: create_valid_job_requirements(), "job_requirements"),
        ],
    )
    def test_node_handles_callback_exception(
        self, mock_llm_provider, large_state, node_name, create_result, result_key
    ):
        """Test that nodes handle callback exceptions gracefully."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        callback = MagicMock(side_effect=Exception("Callback error"))
        expected = create_result()

        with patch.object(RLMOrchestrator, "complete") as mock_complete:
            mock_complete.return_value = create_rlm_result(
                answer=expected, sub_call_count=1, total_iterations=2
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config, on_step_start=callback)
            result = nodes[node_name](large_state)

            callback.assert_called_once()
            assert result[result_key] == expected


# =============================================================================
# Chunking Tests
# =============================================================================


class TestCVChunkerEdgeCases:
    """Edge case tests for CV chunker."""

    @pytest.mark.parametrize(
        "section_name,content",
        [
            ("contact", "john@example.com"),
            ("certifications", "AWS Certified"),
        ],
    )
    def test_assign_section(self, section_name, content):
        """Test assigning various sections."""
        chunker = CVChunker()
        chunks = CVChunks(raw_text="test")
        chunker._assign_section(chunks, section_name, content)
        assert getattr(chunks, section_name) == content

    def test_split_experiences_by_dates(self):
        """Test splitting experiences by date patterns."""
        chunker = CVChunker()
        experience_text = """Senior Developer
2020 - Present
Did great things

Junior Developer
2015 - 2020
Learned a lot"""
        entries = chunker._split_experiences(experience_text)
        assert len(entries) >= 1

    def test_split_experiences_single_entry(self):
        """Test splitting experiences with single entry."""
        chunker = CVChunker()
        experience_text = "Just one job with no clear separation"
        entries = chunker._split_experiences(experience_text)
        assert len(entries) == 1
        assert entries[0] == experience_text

    def test_chunk_with_all_sections(self):
        """Test chunking CV with all section types."""
        cv_text = """# John Doe

## Contact Information
john@example.com

## Summary
Experienced developer

## Experience
### Dev at Company
2020 - Present

## Education
### BS CS
University, 2015

## Skills
Python, Java

## Certifications
AWS Certified

## Languages
English, Spanish

## Awards
Best Employee 2020

## Publications
Paper on AI
"""
        chunker = CVChunker()
        chunks = chunker.chunk(cv_text)

        assert chunks.summary is not None
        assert chunks.skills is not None
        assert len(chunks.sections) > 0

    def test_get_section_from_dict(self):
        """Test getting section from sections dict."""
        chunker = CVChunker()
        chunks = CVChunks(raw_text="test", sections={"custom section": "Custom content"})
        result = chunker.get_section(chunks, "custom section")
        assert result == "Custom content"

    def test_empty_experience_section(self):
        """Test handling empty experience section."""
        chunker = CVChunker()
        cv_empty_exp = """# Name

## Summary
A summary.

## Experience

## Skills
Python, JavaScript
"""
        chunks = chunker.chunk(cv_empty_exp)
        assert chunks.sections is not None


class TestJobChunkerEdgeCases:
    """Edge case tests for job chunker."""

    def test_chunk_no_headers(self):
        """Test chunking job with no headers."""
        chunker = JobChunker()
        job_text = "Just a plain text job posting with no markdown headers."
        chunks = chunker.chunk(job_text)
        assert chunks.overview == job_text.strip()

    def test_chunk_title_from_first_line(self):
        """Test extracting title from first line without header."""
        chunker = JobChunker()
        job_text = """Software Engineer Position

## Requirements
- Python
"""
        chunks = chunker.chunk(job_text)
        assert "Software Engineer" in chunks.title

    def test_extract_bullets_no_bullets(self):
        """Test bullet extraction with no bullet patterns."""
        chunker = JobChunker()
        text = """Short line
Another short one
This is a longer line that should be included in results
And another longer line for the requirements section"""
        bullets = chunker._extract_bullets(text)
        assert len(bullets) >= 1

    @pytest.mark.parametrize(
        "required,expected_min_count",
        [
            (True, 2),  # Must have, Required, and unclassified
            (False, 2),  # Nice to have and Bonus
        ],
    )
    def test_classify_requirements(self, required, expected_min_count):
        """Test classifying requirements."""
        chunker = JobChunker()
        bullets = [
            "Must have 5 years experience",
            "Required: Python knowledge",
            "Nice to have: Docker",
            "Bonus: ML experience",
        ]
        result = chunker._classify_requirements(bullets, required=required)
        assert len(result) >= expected_min_count

    def test_get_requirements_by_priority_with_top_n(self):
        """Test get_requirements_by_priority with top_n limit."""
        chunker = JobChunker()
        chunks = JobChunks(
            raw_text="test",
            required_qualifications=["Req1", "Req2", "Req3"],
            preferred_qualifications=["Pref1", "Pref2"],
        )
        reqs = chunker.get_requirements_by_priority(chunks, top_n=3)
        assert len(reqs) == 3


# =============================================================================
# Environment Tests
# =============================================================================


class TestREPLEnvironmentAdditional:
    """Additional tests for REPL environment."""

    @pytest.mark.parametrize(
        "sandbox_mode,should_succeed",
        [
            ("docker", True),  # Falls back to local
            ("modal", True),  # Falls back to local
        ],
    )
    def test_execute_sandbox_modes(self, sandbox_mode, should_succeed):
        """Test execute in various sandbox modes."""
        env = REPLEnvironment(cv_text="CV", job_text="Job", sandbox_mode=sandbox_mode)
        result = env.execute("print('hello')")
        assert result.success == should_succeed
        if should_succeed:
            assert "hello" in result.output

    def test_execute_unknown_mode(self):
        """Test execute with unknown sandbox mode."""
        env = REPLEnvironment(cv_text="CV", job_text="Job")
        env.sandbox_mode = "unknown"
        result = env.execute("print('hello')")
        assert not result.success
        assert "Unknown sandbox mode" in result.error

    def test_context_summary_with_chunks(self):
        """Test context summary with CV and job chunks."""
        cv_chunks = CVChunks(
            raw_text="CV",
            sections={"skills": "Python"},
            experiences=["Job 1", "Job 2"],
            education=["Degree 1"],
        )
        job_chunks = JobChunks(
            raw_text="Job",
            sections={"requirements": "Skills"},
            required_qualifications=["Python", "Java"],
            preferred_qualifications=["Kubernetes"],
        )

        env = REPLEnvironment(
            cv_text="Full CV text",
            job_text="Full job text",
            cv_chunks=cv_chunks,
            job_chunks=job_chunks,
        )
        summary = env.get_context_summary()

        assert "cv_sections: 1 sections" in summary
        assert "cv_experiences: 2 items" in summary
        assert "job_requirements: 2 items" in summary

    def test_context_summary_with_variables_and_functions(self):
        """Test context summary with stored variables and functions."""
        env = REPLEnvironment(cv_text="CV", job_text="Job")
        env.set_variable("my_result", "A" * 200)
        env.register_function("helper", lambda x: x)
        summary = env.get_context_summary()

        assert "Stored Results:" in summary
        assert "my_result:" in summary
        assert "..." in summary
        assert "Available Functions:" in summary
        assert "helper()" in summary

    def test_execute_output_truncation(self):
        """Test execute truncates long output."""
        env = REPLEnvironment(cv_text="CV", job_text="Job", max_output_length=100)
        result = env.execute("print('A' * 500)")
        assert result.success
        assert "truncated" in result.output
        assert len(result.output) <= 150

    def test_execute_tracks_new_variables(self):
        """Test that execute tracks new variables correctly."""
        env = REPLEnvironment(cv_text="CV", job_text="Job")
        result = env.execute("new_var = 42\nanother_var = 'hello'")
        assert result.success
        assert "new_var" in env.variables
        assert "another_var" in env.variables
        assert env.variables["new_var"] == 42

    @pytest.mark.parametrize(
        "blocked_code",
        [
            "import os",
            "import sys",
            "import subprocess",
            "from os import path",
            "__import__('os')",
            "exec('code')",
            "eval('1+1')",
            "open('file.txt')",
            "compile('code', '', 'exec')",
            "globals()",
            "locals()",
            "getattr(obj, 'attr')",
            "__builtins__",
            "__class__",
            "__bases__",
            "__subclasses__",
        ],
    )
    def test_validate_code_blocked_patterns(self, blocked_code):
        """Test all blocked patterns are detected."""
        env = REPLEnvironment(cv_text="CV", job_text="Job")
        is_safe, error = env._validate_code(blocked_code)
        assert not is_safe, f"Should block: {blocked_code}"
        assert error is not None


# =============================================================================
# Is Valid CV Data Tests
# =============================================================================


class TestIsValidCVData:
    """Tests for _is_valid_cv_data helper function."""

    @pytest.mark.parametrize(
        "name,experiences,skills,expected",
        [
            ("John Doe", [Experience(title="Dev", company="Co", start_date="2020")], [], True),
            ("Jane Doe", [], ["Python", "JS"], True),
            ("UNKNOWN", [], ["Python"], False),
            ("<UNKNOWN>", [], ["Python"], False),
            ("N/A", [], ["Python"], False),
            ("", [], ["Python"], False),
            ("NAME NOT PROVIDED", [], ["Python"], False),
            ("John Doe", [], [], False),  # No experiences or skills
        ],
    )
    def test_is_valid_cv_data(self, name, experiences, skills, expected):
        """Test _is_valid_cv_data with various inputs."""
        from cv_warlock.graph.rlm_nodes import _is_valid_cv_data

        cv_data = CVData(
            contact=ContactInfo(name=name, email="test@test.com"),
            summary="Test",
            experiences=experiences,
            education=[],
            skills=skills,
        )
        assert _is_valid_cv_data(cv_data) is expected

    @pytest.mark.parametrize("invalid_input", ["not a CVData", {"name": "test"}, None])
    def test_is_valid_cv_data_wrong_type(self, invalid_input):
        """Test _is_valid_cv_data with non-CVData types."""
        from cv_warlock.graph.rlm_nodes import _is_valid_cv_data

        assert _is_valid_cv_data(invalid_input) is False


# =============================================================================
# Timeout Tests
# =============================================================================


class TestRLMTimeoutHelpers:
    """Tests for RLM timeout helper functions."""

    def test_check_timeout_true(self):
        """Test _check_rlm_timeout returns True for timeout."""
        from cv_warlock.graph.rlm_nodes import _check_rlm_timeout

        rlm_result = create_rlm_result(
            answer=None, trajectory=[create_timeout_step()], success=False
        )
        assert _check_rlm_timeout(rlm_result) is True

    def test_check_timeout_false(self):
        """Test _check_rlm_timeout returns False when no timeout."""
        from cv_warlock.graph.rlm_nodes import _check_rlm_timeout

        step = TrajectoryStep(
            step_number=1,
            action_type=ActionType.FINAL,
            model_output="FINAL(answer)",
            parsed_action=None,
            execution_result="Done",
            duration_ms=100,
        )
        rlm_result = create_rlm_result(answer="answer", trajectory=[step])
        assert _check_rlm_timeout(rlm_result) is False

    def test_check_timeout_empty_trajectory(self):
        """Test _check_rlm_timeout returns False for empty trajectory."""
        from cv_warlock.graph.rlm_nodes import _check_rlm_timeout

        rlm_result = create_rlm_result(answer=None, trajectory=[], success=False)
        assert _check_rlm_timeout(rlm_result) is False

    def test_get_timeout_message(self):
        """Test _get_timeout_message returns message when timeout."""
        from cv_warlock.graph.rlm_nodes import _get_timeout_message

        rlm_result = create_rlm_result(
            answer=None, trajectory=[create_timeout_step()], success=False
        )
        assert _get_timeout_message(rlm_result) == "RLM timeout after 30s"

    def test_get_timeout_message_no_timeout(self):
        """Test _get_timeout_message returns None when no timeout."""
        from cv_warlock.graph.rlm_nodes import _get_timeout_message

        step = TrajectoryStep(
            step_number=1,
            action_type=ActionType.FINAL,
            model_output="Normal",
            parsed_action=None,
            execution_result="Done",
            duration_ms=100,
        )
        rlm_result = create_rlm_result(answer="answer", trajectory=[step])
        assert _get_timeout_message(rlm_result) is None


class TestExtractionTimeouts:
    """Tests for extraction node timeout handling."""

    def test_extract_cv_timeout_with_successful_fallback(self, mock_llm_provider, large_state):
        """Test extract_cv timeout with successful fallback."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        fallback_result = create_valid_cv_data("Fallback")

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create,
        ):
            mock_complete.return_value = create_rlm_result(
                answer=None, trajectory=[create_timeout_step()], success=False
            )
            mock_create.return_value = create_mock_standard_nodes(
                extract_cv={"cv_data": fallback_result, "current_step": "extract_cv"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_cv"](large_state)

            assert result["cv_data"] == fallback_result
            assert result["rlm_metadata"]["used"] is False

    def test_extract_job_timeout_with_successful_fallback(self, mock_llm_provider, large_state):
        """Test extract_job timeout with successful fallback."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        fallback_result = create_valid_job_requirements("Fallback")

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create,
        ):
            mock_complete.return_value = create_rlm_result(
                answer=None, trajectory=[create_timeout_step()], success=False
            )
            mock_create.return_value = create_mock_standard_nodes(
                extract_job={"job_requirements": fallback_result, "current_step": "extract_job"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_job"](large_state)

            assert result["job_requirements"] == fallback_result

    @pytest.mark.parametrize(
        "node_name,result_key",
        [
            ("extract_cv", "cv_data"),
            ("extract_job", "job_requirements"),
        ],
    )
    def test_timeout_with_failed_fallback(
        self, mock_llm_provider, large_state, node_name, result_key
    ):
        """Test extraction timeout with failed fallback adds error."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create,
        ):
            mock_complete.return_value = create_rlm_result(
                answer=None, trajectory=[create_timeout_step()], success=False
            )
            mock_create.return_value = create_mock_standard_nodes(
                **{node_name: {result_key: None, "current_step": node_name}}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes[node_name](large_state)

            assert result[result_key] is None
            assert "errors" in result
            assert any("timeout" in str(e).lower() for e in result["errors"])


class TestAnalyzeMatchTimeout:
    """Tests for analyze_match_rlm timeout handling."""

    def test_analyze_match_timeout_fallback(self, mock_llm_provider, large_state):
        """Test analyze_match_rlm timeout falls back to standard."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create,
        ):
            mock_complete.return_value = create_rlm_result(
                answer=None, trajectory=[create_timeout_step()], success=False
            )
            mock_create.return_value = create_mock_standard_nodes(
                analyze_match=create_match_analysis_result(0.7)
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["analyze_match"](large_state)

            assert result["match_analysis"]["relevance_score"] == 0.7


# =============================================================================
# LLM Extraction Tests
# =============================================================================


class TestExtractWithLLM:
    """Tests for _extract_with_llm method coverage."""

    def test_extract_with_llm_success(self, sample_cv_text, sample_job_text):
        """Test successful LLM extraction."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        mock_extraction_model = MagicMock()
        mock_structured_model = MagicMock()
        mock_config = RLMConfig(max_iterations=5)
        mock_structured_model.invoke.return_value = mock_config
        mock_extraction_model.with_structured_output.return_value = mock_structured_model
        sub_provider.get_extraction_model.return_value = mock_extraction_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        env.set_variable("analysis_key", {"some": "data"})

        content = "A" * 200 + " analysis content with meaningful data"
        result = orchestrator._extract_with_llm(content, RLMConfig, env)

        assert isinstance(result, RLMConfig)

    def test_extract_with_llm_exception_returns_content(self, sample_cv_text, sample_job_text):
        """Test LLM extraction returns original content on exception."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        mock_extraction_model = MagicMock()
        mock_extraction_model.with_structured_output.side_effect = Exception("LLM error")
        sub_provider.get_extraction_model.return_value = mock_extraction_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)

        result = orchestrator._extract_with_llm("some content", RLMConfig, env)
        assert result == "some content"


class TestOrchestratorCodeAndFinal:
    """Tests for code execution followed by FINAL."""

    def test_code_then_final_in_same_response(self, sample_cv_text, sample_job_text):
        """Test code block followed by FINAL() in same response."""
        provider = MagicMock()
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(
            content="""Let me analyze.

```python
result = {"skills": ["Python", "Java"]}
print("Analyzed!")
```

FINAL(The candidate has strong Python and Java skills)"""
        )
        provider.get_chat_model.return_value = mock_model

        config = RLMConfig(max_iterations=5)
        orchestrator = RLMOrchestrator(provider, config=config)
        result = orchestrator.complete(
            task="Analyze skills",
            cv_text=sample_cv_text,
            job_text=sample_job_text,
        )

        assert result.success

    def test_code_then_final_var_in_same_response(self, sample_cv_text, sample_job_text):
        """Test code block followed by FINAL_VAR() in same response."""
        provider = MagicMock()
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(
            content="""```python
analysis_result = "Candidate is a great match with 5+ years Python experience"
print("Done!")
```

FINAL_VAR(analysis_result)"""
        )
        provider.get_chat_model.return_value = mock_model

        config = RLMConfig(max_iterations=5)
        orchestrator = RLMOrchestrator(provider, config=config)
        result = orchestrator.complete(
            task="Analyze match",
            cv_text=sample_cv_text,
            job_text=sample_job_text,
        )

        assert result.success
        assert result.total_iterations == 1


class TestOrchestratorMaxSubCalls:
    """Tests for max sub-calls handling."""

    def test_max_sub_calls_reached(self, sample_cv_text, sample_job_text):
        """Test feedback when max sub-calls reached."""
        root_provider = MagicMock()
        sub_provider = MagicMock()
        root_model = MagicMock()
        sub_model = MagicMock()

        call_count = [0]

        def mock_root_invoke(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:
                return MagicMock(content='```python\nresult = rlm_query(cv_text, "Q?")\n```')
            return MagicMock(content="FINAL(Done after max sub-calls)")

        root_model.invoke.side_effect = mock_root_invoke
        root_provider.get_chat_model.return_value = root_model
        sub_model.invoke.return_value = MagicMock(content="Sub answer")
        sub_provider.get_chat_model.return_value = sub_model

        config = RLMConfig(max_iterations=10, max_sub_calls=2)
        orchestrator = RLMOrchestrator(root_provider, sub_provider, config=config)
        result = orchestrator.complete(
            task="Analyze",
            cv_text=sample_cv_text,
            job_text=sample_job_text,
        )

        assert result.sub_call_count <= 2


class TestOrchestratorFallbackExtraction:
    """Tests for fallback answer extraction edge cases."""

    def test_fallback_with_invalid_json_triggers_llm(self, sample_cv_text, sample_job_text):
        """Test fallback extraction with invalid JSON triggers LLM extraction."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        mock_extraction_model = MagicMock()
        mock_structured_model = MagicMock()
        mock_config = RLMConfig(max_iterations=3)
        mock_structured_model.invoke.return_value = mock_config
        mock_extraction_model.with_structured_output.return_value = mock_structured_model
        sub_provider.get_extraction_model.return_value = mock_extraction_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        env.variables["result"] = "not valid json at all"

        answer = orchestrator._extract_fallback_answer(env, RLMConfig)
        assert isinstance(answer, RLMConfig)


class TestExtractCVInvalidData:
    """Test extract_cv_rlm with invalid CVData falls back."""

    def test_invalid_cvdata_triggers_fallback(self, mock_llm_provider, large_state):
        """Test extract_cv_rlm with invalid CVData falls back."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        invalid_cv = CVData(
            contact=ContactInfo(name="UNKNOWN", email=""),
            summary="",
            experiences=[],
            education=[],
            skills=[],
        )
        valid_cv = create_valid_cv_data("Real Name")

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create,
        ):
            mock_complete.return_value = create_rlm_result(answer=invalid_cv)
            mock_create.return_value = create_mock_standard_nodes(
                extract_cv={"cv_data": valid_cv, "current_step": "extract_cv"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_cv"](large_state)

            assert result["cv_data"] == valid_cv
            assert result["rlm_metadata"]["used"] is False


class TestExtractJobMetadataCombine:
    """Test extract_job_rlm combines metadata correctly."""

    def test_combines_existing_metadata(self, mock_llm_provider, large_state):
        """Test extract_job_rlm combines existing RLM metadata."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        state = {**large_state, "rlm_metadata": {"total_iterations": 2, "sub_call_count": 1}}
        job_requirements = create_valid_job_requirements()

        with patch.object(RLMOrchestrator, "complete") as mock_complete:
            mock_complete.return_value = create_rlm_result(
                answer=job_requirements, sub_call_count=2, total_iterations=3
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_job"](state)

            assert result["job_requirements"] == job_requirements
            assert result["rlm_metadata"]["total_iterations"] == 5  # 2 + 3
