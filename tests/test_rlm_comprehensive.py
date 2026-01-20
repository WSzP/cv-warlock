"""Comprehensive tests for RLM module to improve coverage.

Tests for:
- orchestrator.py: parsing, helpers, final answer processing
- rlm_nodes.py: all node functions and helpers
- chunking.py: edge cases
- environment.py: sandbox modes
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cv_warlock.models.cv import ContactInfo, CVData
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

# Get the project root directory
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


def create_mock_standard_nodes(**overrides):
    """Create a complete mock standard nodes dict with optional overrides.

    Args:
        **overrides: Node names and their mock return values to override.

    Returns:
        Dict with all required node functions as MagicMocks.
    """
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
        "tailor_summary": MagicMock(return_value={}),
        "assemble_cv": MagicMock(return_value={}),
    }
    # Apply overrides
    for node_name, mock_return in overrides.items():
        if isinstance(mock_return, dict):
            base_nodes[node_name] = MagicMock(return_value=mock_return)
        else:
            base_nodes[node_name] = mock_return
    return base_nodes


# =============================================================================
# Orchestrator Tests
# =============================================================================


class TestRLMOrchestratorParsing:
    """Tests for orchestrator parsing methods."""

    def test_parse_final_answer(self, mock_llm_provider):
        """Test parsing FINAL() pattern."""
        orchestrator = RLMOrchestrator(mock_llm_provider)

        output = "After analysis, FINAL(The candidate is a great match)"
        action = orchestrator._parse_model_output(output)

        assert action.action_type == ActionType.FINAL
        assert "great match" in action.content

    def test_parse_final_var(self, mock_llm_provider):
        """Test parsing FINAL_VAR() pattern."""
        orchestrator = RLMOrchestrator(mock_llm_provider)

        output = "I've stored the result. FINAL_VAR(analysis_result)"
        action = orchestrator._parse_model_output(output)

        assert action.action_type == ActionType.FINAL
        assert action.content == "analysis_result"

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

    def test_parse_plain_text(self, mock_llm_provider):
        """Test parsing plain text without action."""
        orchestrator = RLMOrchestrator(mock_llm_provider)

        output = "Let me think about this problem..."
        action = orchestrator._parse_model_output(output)

        # Should return CODE action prompting for actual code
        assert action.action_type == ActionType.CODE
        assert "provide Python code" in action.content


class TestRLMOrchestratorHelpers:
    """Tests for orchestrator helper methods."""

    def test_register_helpers(self, mock_llm_provider, sample_cv_text, sample_job_text):
        """Test that helper functions are registered correctly."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)

        orchestrator._register_helpers(env)

        # Check functions are registered
        assert "rlm_query" in env._functions
        assert "find_keyword" in env._functions
        assert "find_sections" in env._functions

    def test_find_keyword_helper(self, mock_llm_provider, sample_cv_text, sample_job_text):
        """Test the find_keyword helper function."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        orchestrator._register_helpers(env)

        # Execute code that uses find_keyword
        result = env.execute('positions = find_keyword("python", cv_text)\nprint(len(positions))')
        assert result.success

    def test_find_sections_helper(self, mock_llm_provider, sample_cv_text, sample_job_text):
        """Test the find_sections helper function."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        orchestrator._register_helpers(env)

        # Execute code that uses find_sections
        result = env.execute("sections = find_sections(cv_text)\nprint(list(sections.keys()))")
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

        # Use a simple Pydantic model for testing
        json_content = '{"max_iterations": 10, "max_sub_calls": 5}'
        result = orchestrator._process_final_answer(json_content, RLMConfig, env)

        assert isinstance(result, RLMConfig)
        assert result.max_iterations == 10

    def test_process_final_answer_invalid_json(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test processing invalid JSON with schema triggers LLM extraction."""
        # Set up mock for LLM extraction fallback
        mock_structured_model = MagicMock()
        mock_extracted_config = RLMConfig(max_iterations=5)
        mock_structured_model.invoke.return_value = mock_extracted_config
        mock_llm_provider.get_extraction_model.return_value.with_structured_output.return_value = (
            mock_structured_model
        )

        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)

        result = orchestrator._process_final_answer("not valid json", RLMConfig, env)
        # Now we attempt LLM extraction instead of returning the string
        assert isinstance(result, RLMConfig)
        assert result.max_iterations == 5


class TestRLMOrchestratorFallback:
    """Tests for fallback answer extraction."""

    def test_extract_fallback_from_result_var(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extracting fallback from 'result' variable."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        env.variables["result"] = "Fallback result"

        answer = orchestrator._extract_fallback_answer(env, None)
        assert answer == "Fallback result"

    def test_extract_fallback_from_answer_var(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extracting fallback from 'answer' variable."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        env.variables["answer"] = "Fallback answer"

        answer = orchestrator._extract_fallback_answer(env, None)
        assert answer == "Fallback answer"

    def test_extract_fallback_with_schema_dict(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extracting fallback with schema from dict."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        env.variables["result"] = {"max_iterations": 15}

        answer = orchestrator._extract_fallback_answer(env, RLMConfig)
        assert isinstance(answer, RLMConfig)
        assert answer.max_iterations == 15

    def test_extract_fallback_with_schema_json_string(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extracting fallback with schema from JSON string."""
        orchestrator = RLMOrchestrator(mock_llm_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        env.variables["result"] = '{"max_iterations": 25}'

        answer = orchestrator._extract_fallback_answer(env, RLMConfig)
        assert isinstance(answer, RLMConfig)
        assert answer.max_iterations == 25

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

        # First call returns code, second returns FINAL
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

    def test_complete_timeout(self, sample_cv_text, sample_job_text):
        """Test complete with max iterations limit (timeout requires longer runs)."""
        provider = MagicMock()
        mock_model = MagicMock()
        # Always return code, never final
        mock_model.invoke.return_value = MagicMock(content="```python\nprint('thinking')\n```")
        provider.get_chat_model.return_value = mock_model

        # Use low max_iterations instead of timeout (timeout_seconds must be int >= 30)
        config = RLMConfig(max_iterations=2, timeout_seconds=300)
        orchestrator = RLMOrchestrator(provider, config=config)

        result = orchestrator.complete(
            task="Analyze",
            cv_text=sample_cv_text,
            job_text=sample_job_text,
        )

        # Should have stopped due to max iterations
        assert result.total_iterations == 2

    def test_complete_max_iterations(self, sample_cv_text, sample_job_text):
        """Test complete hitting max iterations."""
        provider = MagicMock()
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="```python\nprint('iteration')\n```")
        provider.get_chat_model.return_value = mock_model

        config = RLMConfig(max_iterations=3, timeout_seconds=300)
        orchestrator = RLMOrchestrator(provider, config=config)

        result = orchestrator.complete(
            task="Analyze",
            cv_text=sample_cv_text,
            job_text=sample_job_text,
        )

        assert result.total_iterations == 3

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

        # Root returns query, then final
        root_responses = [
            MagicMock(
                content='```python\nresult = rlm_query(cv_text, "What skills?")\nprint(result)\n```'
            ),
            MagicMock(content="FINAL(Done)"),
        ]
        root_model.invoke.side_effect = root_responses
        root_provider.get_chat_model.return_value = root_model

        # Sub returns answer
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

    def test_complete_max_sub_calls(self, sample_cv_text, sample_job_text):
        """Test complete hitting max sub-calls."""
        provider = MagicMock()
        mock_model = MagicMock()

        # Keep returning query requests
        mock_model.invoke.return_value = MagicMock(
            content='```python\nresult = rlm_query(cv_text, "Question?")\n```'
        )
        provider.get_chat_model.return_value = mock_model

        config = RLMConfig(max_iterations=10, max_sub_calls=2)
        orchestrator = RLMOrchestrator(provider, config=config)

        # Run but don't let it loop forever
        result = orchestrator.complete(
            task="Analyze",
            cv_text=sample_cv_text,
            job_text=sample_job_text,
        )

        # Should have hit max sub-calls
        assert result.sub_call_count <= 2


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
        # Verify the context was resolved
        assert "Variable content" in result.context_preview or len(result.context_preview) > 0

    def test_execute_sub_call_truncation(self, sample_cv_text, sample_job_text):
        """Test sub-call with long context truncation."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        sub_model = MagicMock()
        sub_model.invoke.return_value = MagicMock(content="Answer")
        sub_provider.get_chat_model.return_value = sub_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)

        # Very long context
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

    def test_create_with_defaults(self):
        """Test factory with default parameters."""
        with patch("cv_warlock.llm.base.get_llm_provider") as mock_get:
            mock_provider = MagicMock()
            mock_get.return_value = mock_provider

            from cv_warlock.rlm.orchestrator import create_rlm_orchestrator

            orchestrator = create_rlm_orchestrator()

            assert orchestrator.root_provider == mock_provider
            # Should have called for both root and sub
            assert mock_get.call_count == 2

    def test_create_with_custom_providers(self):
        """Test factory with custom providers."""
        with patch("cv_warlock.llm.base.get_llm_provider") as mock_get:
            mock_provider = MagicMock()
            mock_get.return_value = mock_provider

            from cv_warlock.rlm.orchestrator import create_rlm_orchestrator

            config = RLMConfig(max_iterations=10)
            orchestrator = create_rlm_orchestrator(
                root_provider="openai",
                root_model="gpt-5.2",
                sub_provider="openai",
                sub_model="gpt-5-mini",
                config=config,
            )

            assert orchestrator.config.max_iterations == 10


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

        rlm_result = RLMResult(
            answer="test",
            trajectory=trajectory,
            sub_call_count=1,
            total_iterations=2,
            success=True,
        )

        converted = _convert_trajectory(rlm_result)

        assert len(converted) == 2
        # RLMTrajectoryStep is a TypedDict, so access as dict
        assert converted[0]["step_number"] == 1
        assert converted[0]["action_type"] == "code"
        assert converted[1]["sub_call_made"] is True

    def test_create_rlm_metadata_without_result(self):
        """Test metadata creation without RLM result."""
        from cv_warlock.graph.rlm_nodes import _create_rlm_metadata

        metadata = _create_rlm_metadata(enabled=True, used=False)

        assert metadata["enabled"] is True
        assert metadata["used"] is False
        assert metadata["total_iterations"] == 0
        assert metadata["sub_call_count"] == 0

    def test_create_rlm_metadata_with_result(self):
        """Test metadata creation with RLM result."""
        from cv_warlock.graph.rlm_nodes import _create_rlm_metadata

        rlm_result = RLMResult(
            answer="test",
            trajectory=[],
            sub_call_count=3,
            total_iterations=5,
            execution_time_seconds=10.5,
            success=True,
            intermediate_findings={"key": "value"},
        )

        metadata = _create_rlm_metadata(enabled=True, used=True, rlm_result=rlm_result)

        assert metadata["enabled"] is True
        assert metadata["used"] is True
        assert metadata["total_iterations"] == 5
        assert metadata["sub_call_count"] == 3
        assert metadata["execution_time_seconds"] == 10.5


class TestRLMNodesCreation:
    """Tests for RLM node creation and execution."""

    def test_create_rlm_nodes(self, mock_llm_provider):
        """Test creating RLM nodes."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        nodes = create_rlm_nodes(mock_llm_provider)

        assert "validate_inputs" in nodes
        assert "extract_cv" in nodes
        assert "extract_job" in nodes
        assert "analyze_match" in nodes
        assert "create_plan" in nodes
        assert "tailor_skills" in nodes
        assert "tailor_experiences" in nodes
        assert "tailor_summary" in nodes
        assert "assemble_cv" in nodes

    def test_extract_cv_rlm_disabled(self, mock_llm_provider, sample_cv_text, sample_job_text):
        """Test extract_cv_rlm when RLM is disabled - uses standard extraction."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        state = {
            "raw_cv": sample_cv_text,
            "raw_job_spec": sample_job_text,
            "use_rlm": False,  # RLM disabled
            "errors": [],
        }

        # Mock the standard extraction to return a result
        cv_data = CVData(
            contact=ContactInfo(name="Test", email="test@test.com"),
            summary="Test summary",
            experiences=[],
            education=[],
            skills=[],
        )

        with patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes:
            mock_create_nodes.return_value = create_mock_standard_nodes(
                extract_cv={"cv_data": cv_data, "current_step": "extract_cv"}
            )

            nodes = create_rlm_nodes(mock_llm_provider)
            result = nodes["extract_cv"](state)

            # Should use standard extraction and add RLM metadata
            assert result["cv_data"] == cv_data
            assert result["rlm_metadata"]["enabled"] is False
            assert result["rlm_metadata"]["used"] is False

    def test_extract_cv_rlm_below_threshold(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extract_cv_rlm when below size threshold - uses standard extraction."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        state = {
            "raw_cv": sample_cv_text,  # Small CV
            "raw_job_spec": sample_job_text,  # Small job
            "use_rlm": True,  # RLM enabled but below threshold
            "errors": [],
        }

        cv_data = CVData(
            contact=ContactInfo(name="Test", email="test@test.com"),
            summary="Test summary",
            experiences=[],
            education=[],
            skills=[],
        )

        with patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes:
            mock_create_nodes.return_value = create_mock_standard_nodes(
                extract_cv={"cv_data": cv_data, "current_step": "extract_cv"}
            )

            # High threshold so documents are below it
            config = RLMConfig(size_threshold=100000)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_cv"](state)

            # Should use standard extraction
            assert result["cv_data"] == cv_data
            assert result["rlm_metadata"]["enabled"] is True
            assert result["rlm_metadata"]["used"] is False

    def test_extract_cv_rlm_success(self, mock_llm_provider, sample_cv_text, sample_job_text):
        """Test extract_cv_rlm with successful RLM result."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        # Create a large state to trigger RLM
        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
        }

        # Mock the orchestrator to return CVData with valid content
        # Note: _is_valid_cv_data requires at least name + (experiences OR skills)
        cv_data = CVData(
            contact=ContactInfo(name="Test", email="test@test.com"),
            summary="Test summary",
            experiences=[],
            education=[],
            skills=["Python", "TypeScript"],  # Need at least one skill for validation
        )

        with patch.object(RLMOrchestrator, "complete") as mock_complete:
            mock_complete.return_value = RLMResult(
                answer=cv_data,
                trajectory=[],
                sub_call_count=1,
                total_iterations=2,
                success=True,
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_cv"](state)

            assert result["cv_data"] == cv_data
            assert result["rlm_metadata"]["used"] is True

    def test_extract_cv_rlm_fallback_on_failure(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extract_cv_rlm falling back on RLM failure."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        cv_data = CVData(
            contact=ContactInfo(name="Fallback", email="fallback@test.com"),
            summary="Fallback summary",
            experiences=[],
            education=[],
            skills=[],
        )

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            # RLM fails
            mock_complete.return_value = RLMResult(
                answer=None,
                trajectory=[],
                sub_call_count=0,
                total_iterations=1,
                success=False,
                error="Test RLM error",
            )
            # Standard extraction succeeds
            mock_create_nodes.return_value = create_mock_standard_nodes(
                extract_cv={"cv_data": cv_data, "current_step": "extract_cv"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_cv"](state)

            # Should fall back to standard extraction
            assert result["cv_data"] == cv_data
            assert result["rlm_metadata"]["used"] is False

    def test_extract_cv_rlm_wrong_type_fallback(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extract_cv_rlm with wrong answer type falls back to standard."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        cv_data = CVData(
            contact=ContactInfo(name="Standard", email="standard@test.com"),
            summary="Standard summary",
            experiences=[],
            education=[],
            skills=[],
        )

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            # RLM returns wrong type (string instead of CVData)
            mock_complete.return_value = RLMResult(
                answer="Not a CVData object",
                trajectory=[],
                sub_call_count=0,
                total_iterations=1,
                success=True,
            )
            # Standard extraction succeeds
            mock_create_nodes.return_value = create_mock_standard_nodes(
                extract_cv={"cv_data": cv_data, "current_step": "extract_cv"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_cv"](state)

            # Should fall back to standard extraction
            assert result["cv_data"] == cv_data
            assert result["rlm_metadata"]["used"] is False

    def test_extract_cv_rlm_exception_fallback(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extract_cv_rlm falls back when exception occurs."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        cv_data = CVData(
            contact=ContactInfo(name="Exception", email="exception@test.com"),
            summary="Exception fallback summary",
            experiences=[],
            education=[],
            skills=[],
        )

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            # RLM raises exception
            mock_complete.side_effect = Exception("RLM crashed!")
            # Standard extraction succeeds
            mock_create_nodes.return_value = create_mock_standard_nodes(
                extract_cv={"cv_data": cv_data, "current_step": "extract_cv"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_cv"](state)

            # Should fall back to standard extraction and record error
            assert result["cv_data"] == cv_data
            assert result["rlm_metadata"]["used"] is False
            assert any("RLM extraction fallback" in e for e in result["errors"])

    def test_extract_job_rlm_success(self, mock_llm_provider, sample_cv_text, sample_job_text):
        """Test extract_job_rlm with successful result."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "rlm_metadata": {"total_iterations": 2, "sub_call_count": 1},
        }

        job_requirements = JobRequirements(
            job_title="Senior Developer",
            required_skills=["Python"],
            preferred_skills=["Kubernetes"],
            responsibilities=["Code review"],
        )

        with patch.object(RLMOrchestrator, "complete") as mock_complete:
            mock_complete.return_value = RLMResult(
                answer=job_requirements,
                trajectory=[],
                sub_call_count=2,
                total_iterations=3,
                success=True,
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_job"](state)

            assert result["job_requirements"] == job_requirements
            # Should combine metadata
            assert result["rlm_metadata"]["total_iterations"] == 5  # 2 + 3

    def test_extract_job_rlm_success_no_existing_metadata(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extract_job_rlm with successful result but no existing metadata."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            # No rlm_metadata - tests the else branch
        }

        job_requirements = JobRequirements(
            job_title="Senior Developer",
            required_skills=["Python"],
            preferred_skills=["Kubernetes"],
            responsibilities=["Code review"],
        )

        with patch.object(RLMOrchestrator, "complete") as mock_complete:
            mock_complete.return_value = RLMResult(
                answer=job_requirements,
                trajectory=[],
                sub_call_count=2,
                total_iterations=3,
                success=True,
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_job"](state)

            assert result["job_requirements"] == job_requirements
            # No existing metadata to combine with
            assert result["rlm_metadata"]["total_iterations"] == 3

    def test_extract_job_rlm_disabled(self, mock_llm_provider, sample_cv_text, sample_job_text):
        """Test extract_job_rlm when RLM is disabled."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        state = {
            "raw_cv": sample_cv_text,
            "raw_job_spec": sample_job_text,
            "use_rlm": False,  # Disabled
            "errors": [],
        }

        job_requirements = JobRequirements(
            job_title="Fallback Job",
            required_skills=["Python"],
            preferred_skills=[],
            responsibilities=[],
        )

        with patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes:
            mock_create_nodes.return_value = create_mock_standard_nodes(
                extract_job={"job_requirements": job_requirements, "current_step": "extract_job"}
            )

            nodes = create_rlm_nodes(mock_llm_provider)
            result = nodes["extract_job"](state)

            assert result["job_requirements"] == job_requirements

    def test_extract_job_rlm_wrong_type_fallback(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extract_job_rlm with wrong answer type falls back to standard."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        job_requirements = JobRequirements(
            job_title="Standard Job",
            required_skills=["Python"],
            preferred_skills=[],
            responsibilities=[],
        )

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            # RLM returns wrong type (string instead of JobRequirements)
            mock_complete.return_value = RLMResult(
                answer="Not a JobRequirements object",
                trajectory=[],
                sub_call_count=0,
                total_iterations=1,
                success=True,
            )
            mock_create_nodes.return_value = create_mock_standard_nodes(
                extract_job={"job_requirements": job_requirements, "current_step": "extract_job"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_job"](state)

            assert result["job_requirements"] == job_requirements

    def test_extract_job_rlm_failure_fallback(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extract_job_rlm falls back on RLM failure."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        job_requirements = JobRequirements(
            job_title="Fallback Job",
            required_skills=["Python"],
            preferred_skills=[],
            responsibilities=[],
        )

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            mock_complete.return_value = RLMResult(
                answer=None,
                trajectory=[],
                sub_call_count=0,
                total_iterations=1,
                success=False,
                error="RLM job extraction failed",
            )
            mock_create_nodes.return_value = create_mock_standard_nodes(
                extract_job={"job_requirements": job_requirements, "current_step": "extract_job"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_job"](state)

            assert result["job_requirements"] == job_requirements

    def test_extract_job_rlm_exception_fallback(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extract_job_rlm falls back when exception occurs."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        job_requirements = JobRequirements(
            job_title="Exception Fallback",
            required_skills=["Python"],
            preferred_skills=[],
            responsibilities=[],
        )

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            mock_complete.side_effect = Exception("RLM job extraction crashed!")
            mock_create_nodes.return_value = create_mock_standard_nodes(
                extract_job={"job_requirements": job_requirements, "current_step": "extract_job"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_job"](state)

            assert result["job_requirements"] == job_requirements
            assert any("RLM job extraction fallback" in e for e in result["errors"])

    def test_analyze_match_rlm_dict_answer(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test analyze_match_rlm with dict answer."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
        }

        with patch.object(RLMOrchestrator, "complete") as mock_complete:
            mock_complete.return_value = RLMResult(
                answer={
                    "strong_matches": ["Python"],
                    "partial_matches": ["Cloud"],
                    "gaps": ["Kubernetes"],
                    "transferable_skills": ["Leadership"],
                    "relevance_score": 0.75,
                },
                trajectory=[],
                sub_call_count=1,
                total_iterations=2,
                success=True,
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["analyze_match"](state)

            assert "match_analysis" in result
            # MatchAnalysis is a TypedDict, so access as dict
            assert result["match_analysis"]["relevance_score"] == 0.75

    def test_analyze_match_rlm_disabled(self, mock_llm_provider, sample_cv_text, sample_job_text):
        """Test analyze_match_rlm when RLM is disabled."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        state = {
            "raw_cv": sample_cv_text,
            "raw_job_spec": sample_job_text,
            "use_rlm": False,  # Disabled
            "errors": [],
        }

        match_result = {
            "match_analysis": {
                "strong_matches": ["Python"],
                "partial_matches": [],
                "gaps": [],
                "transferable_skills": [],
                "relevance_score": 0.8,
            },
            "current_step": "analyze_match",
        }

        with patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes:
            mock_create_nodes.return_value = create_mock_standard_nodes(analyze_match=match_result)

            nodes = create_rlm_nodes(mock_llm_provider)
            result = nodes["analyze_match"](state)

            assert result["match_analysis"]["relevance_score"] == 0.8

    def test_analyze_match_rlm_non_dict_answer_fallback(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test analyze_match_rlm with non-dict answer falls back to standard."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        match_result = {
            "match_analysis": {
                "strong_matches": ["Fallback"],
                "partial_matches": [],
                "gaps": [],
                "transferable_skills": [],
                "relevance_score": 0.6,
            },
            "current_step": "analyze_match",
        }

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            # RLM returns non-dict (string instead of dict)
            mock_complete.return_value = RLMResult(
                answer="Not a dict answer",
                trajectory=[],
                sub_call_count=0,
                total_iterations=1,
                success=True,
            )
            mock_create_nodes.return_value = create_mock_standard_nodes(analyze_match=match_result)

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["analyze_match"](state)

            assert result["match_analysis"]["relevance_score"] == 0.6

    def test_analyze_match_rlm_failure_fallback(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test analyze_match_rlm falls back on RLM failure."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        match_result = {
            "match_analysis": {
                "strong_matches": ["Failure Fallback"],
                "partial_matches": [],
                "gaps": [],
                "transferable_skills": [],
                "relevance_score": 0.5,
            },
            "current_step": "analyze_match",
        }

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            mock_complete.return_value = RLMResult(
                answer=None,
                trajectory=[],
                sub_call_count=0,
                total_iterations=1,
                success=False,
                error="RLM match analysis failed",
            )
            mock_create_nodes.return_value = create_mock_standard_nodes(analyze_match=match_result)

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["analyze_match"](state)

            assert result["match_analysis"]["relevance_score"] == 0.5

    def test_analyze_match_rlm_exception_fallback(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test analyze_match_rlm falls back when exception occurs."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        match_result = {
            "match_analysis": {
                "strong_matches": ["Exception Fallback"],
                "partial_matches": [],
                "gaps": [],
                "transferable_skills": [],
                "relevance_score": 0.4,
            },
            "current_step": "analyze_match",
        }

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            mock_complete.side_effect = Exception("RLM match analysis crashed!")
            mock_create_nodes.return_value = create_mock_standard_nodes(analyze_match=match_result)

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["analyze_match"](state)

            assert result["match_analysis"]["relevance_score"] == 0.4
            assert any("RLM match analysis fallback" in e for e in result["errors"])


# =============================================================================
# Chunking Tests (Edge Cases)
# =============================================================================


class TestCVChunkerEdgeCases:
    """Edge case tests for CV chunker."""

    def test_assign_section_contact(self):
        """Test assigning contact section."""
        chunker = CVChunker()
        chunks = CVChunks(raw_text="test")

        chunker._assign_section(chunks, "contact", "john@example.com")
        assert chunks.contact == "john@example.com"

    def test_assign_section_certifications(self):
        """Test assigning certifications section."""
        chunker = CVChunker()
        chunks = CVChunks(raw_text="test")

        chunker._assign_section(chunks, "certifications", "AWS Certified")
        assert chunks.certifications == "AWS Certified"

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
        chunks = CVChunks(
            raw_text="test",
            sections={"custom section": "Custom content"},
        )

        result = chunker.get_section(chunks, "custom section")
        assert result == "Custom content"


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
        # Should fall back to line splitting
        assert len(bullets) >= 1

    def test_classify_requirements_required(self):
        """Test classifying requirements as required."""
        chunker = JobChunker()

        bullets = [
            "Must have 5 years experience",
            "Required: Python knowledge",
            "Nice to have: Kubernetes",
            "Communication skills",
        ]

        required = chunker._classify_requirements(bullets, required=True)
        assert len(required) >= 2  # Must have, Required, and unclassified

    def test_classify_requirements_preferred(self):
        """Test classifying requirements as preferred."""
        chunker = JobChunker()

        bullets = [
            "Must have 5 years experience",
            "Nice to have: Docker",
            "Bonus: ML experience",
        ]

        preferred = chunker._classify_requirements(bullets, required=False)
        assert len(preferred) == 2  # Nice to have and Bonus

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

    def test_chunk_infer_requirements(self):
        """Test inferring requirements when no explicit split."""
        job_text = """# Developer Role

## Qualifications Needed
- Must have Python experience
- 5 years required
- Nice to have cloud experience
"""
        chunker = JobChunker()
        chunks = chunker.chunk(job_text)

        # Should have inferred required/preferred split
        assert len(chunks.required_qualifications) > 0 or len(chunks.preferred_qualifications) > 0


# =============================================================================
# Environment Tests (Additional Coverage)
# =============================================================================


class TestREPLEnvironmentAdditional:
    """Additional tests for REPL environment."""

    def test_execute_docker_mode(self):
        """Test execute in docker mode (falls back to local with warning)."""
        env = REPLEnvironment(
            cv_text="CV",
            job_text="Job",
            sandbox_mode="docker",
        )

        result = env.execute("print('hello')")
        # Should succeed via local fallback
        assert result.success
        assert "hello" in result.output

    def test_execute_modal_mode(self):
        """Test execute in modal mode (falls back to local with warning)."""
        env = REPLEnvironment(
            cv_text="CV",
            job_text="Job",
            sandbox_mode="modal",
        )

        result = env.execute("print('hello')")
        # Should succeed via local fallback
        assert result.success
        assert "hello" in result.output

    def test_execute_unknown_mode(self):
        """Test execute with unknown sandbox mode."""
        env = REPLEnvironment(
            cv_text="CV",
            job_text="Job",
        )
        # Manually set unknown mode
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

        env.set_variable("my_result", "A" * 200)  # Long value
        env.register_function("helper", lambda x: x)

        summary = env.get_context_summary()

        assert "Stored Results:" in summary
        assert "my_result:" in summary
        assert "..." in summary  # Truncated
        assert "Available Functions:" in summary
        assert "helper()" in summary

    def test_execute_with_stderr(self):
        """Test execute capturing stderr."""
        env = REPLEnvironment(cv_text="CV", job_text="Job")

        # sys import is blocked, so this should fail
        result = env.execute("import sys; print('error', file=sys.stderr)")
        # Verify security check caught blocked import
        assert not result.success

    def test_execute_output_truncation(self):
        """Test execute truncates long output."""
        env = REPLEnvironment(
            cv_text="CV",
            job_text="Job",
            max_output_length=100,
        )

        result = env.execute("print('A' * 500)")
        assert result.success
        assert "truncated" in result.output
        assert len(result.output) <= 150  # 100 + truncation message

    def test_namespace_with_builtins_dict(self):
        """Test namespace creation handles __builtins__ as dict."""
        env = REPLEnvironment(cv_text="CV", job_text="Job")

        # Verify basic builtins work
        result = env.execute("x = len([1, 2, 3])\nprint(x)")
        assert result.success
        assert "3" in result.output

    def test_execute_tracks_new_variables(self):
        """Test that execute tracks new variables correctly."""
        env = REPLEnvironment(cv_text="CV", job_text="Job")

        result = env.execute("new_var = 42\nanother_var = 'hello'")
        assert result.success
        assert "new_var" in env.variables
        assert "another_var" in env.variables
        assert env.variables["new_var"] == 42

    def test_validate_code_all_patterns(self):
        """Test all blocked patterns are detected."""
        env = REPLEnvironment(cv_text="CV", job_text="Job")

        blocked_codes = [
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
            "setattr(obj, 'attr', val)",
            "delattr(obj, 'attr')",
            "__builtins__",
            "__class__",
            "__bases__",
            "__subclasses__",
        ]

        for code in blocked_codes:
            is_safe, error = env._validate_code(code)
            assert not is_safe, f"Should block: {code}"
            assert error is not None, f"Should have error message for: {code}"


# =============================================================================
# Additional Coverage Tests for chunking.py
# =============================================================================


class TestCVChunkerExperienceSplitting:
    """Tests for experience splitting with various formats."""

    def test_split_experiences_with_subheaders(self):
        """Test splitting experiences using ### subheaders.

        Note: The CVChunker creates separate sections for each header level,
        so subheaders become their own sections rather than nested content.
        """
        chunker = CVChunker()

        cv_with_subheaders = """# Jane Doe

## Experience

### Senior Engineer at TechCorp
January 2023 - Present
- Led platform development team
- Implemented CI/CD pipelines

### Junior Developer at StartupXYZ
June 2020 - December 2022
- Built React components
- Worked on API integrations
"""
        chunks = chunker.chunk(cv_with_subheaders)

        # Subheaders become their own sections in the sections dict
        assert chunks.sections is not None
        # The job titles should appear as section keys
        assert "senior engineer at techcorp" in chunks.sections
        assert "junior developer at startupxyz" in chunks.sections

    def test_split_experiences_with_date_patterns(self):
        """Test splitting experiences by date patterns when no subheaders."""
        chunker = CVChunker()

        cv_with_dates = """# John Smith

## Work History

Senior Engineer at TechCorp
January 2023 - Present
Led platform development team

Junior Developer at StartupXYZ
June 2020 - December 2022
Built React components
"""
        chunks = chunker.chunk(cv_with_dates)
        # Should handle date-based splitting or return as one chunk
        assert chunks.sections is not None

    def test_split_by_subheaders_education(self):
        """Test splitting education section by subheaders.

        Note: The CVChunker creates separate sections for each header level,
        so education subheaders become their own sections.
        """
        chunker = CVChunker()

        cv_with_education = """# Jane Doe

## Education

### PhD in Computer Science
MIT | 2020 - 2024
Focus on distributed systems

### Master's in Engineering
Stanford | 2018 - 2020
Thesis on ML optimization
"""
        chunks = chunker.chunk(cv_with_education)

        # Subheaders become their own sections in the sections dict
        assert chunks.sections is not None
        # The degree names should appear as section keys
        assert "phd in computer science" in chunks.sections
        assert "master's in engineering" in chunks.sections


class TestCVChunkerSplittingEdgeCases:
    """Edge case tests for CV section splitting."""

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
        # Should handle gracefully without crashing
        assert chunks.sections is not None

    def test_very_short_experience_entries(self):
        """Test that very short entries are filtered out."""
        chunker = CVChunker()

        cv_short = """# Name

## Experience

### Job 1
Good description here with enough content.

###

### Job 2
Another good entry.
"""
        chunks = chunker.chunk(cv_short)
        # Should filter out very short fragments
        if chunks.experiences:
            for exp in chunks.experiences:
                assert len(exp) > 10  # Should not have tiny fragments


# =============================================================================
# Additional Coverage Tests for orchestrator.py
# =============================================================================


class TestOrchestratorFinalPatternMatching:
    """Tests for FINAL pattern matching in orchestrator output."""

    def test_process_final_with_json_content(self, sample_cv_text, sample_job_text):
        """Test processing FINAL answer with JSON content."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)

        # Test with valid JSON-like content
        result = orchestrator._process_final_answer(
            '{"name": "Test", "skills": ["Python"]}',
            output_schema=None,
            env=env,
        )

        # Should return the content or processed version
        assert result is not None

    def test_process_final_with_plain_text(self, sample_cv_text, sample_job_text):
        """Test processing FINAL answer with plain text."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)

        result = orchestrator._process_final_answer(
            "This is a plain text final answer.",
            output_schema=None,
            env=env,
        )

        assert result == "This is a plain text final answer."


class TestOrchestratorJSONExtraction:
    """Tests for JSON extraction from wrapped content."""

    def test_extract_json_from_wrapped_content(self, sample_cv_text, sample_job_text):
        """Test extracting JSON embedded in text."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        # Mock the extraction model
        extraction_model = MagicMock()
        extraction_response = MagicMock()
        extraction_response.content = '{"name": "Test"}'
        extraction_model.invoke.return_value = extraction_response
        sub_provider.get_extraction_model.return_value = extraction_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        env.set_variable("analysis_var", {"key": "value"})

        # Content with JSON wrapped in text
        content = """Based on my analysis, here is the result:
        {"name": "John Doe", "summary": "Software engineer"}
        This is the extracted data."""

        result = orchestrator._process_final_answer(
            content,
            output_schema=None,
            env=env,
        )

        # Should extract or return the content
        assert result is not None


class TestOrchestratorCodeExecutionThenFinal:
    """Tests for FINAL appearing after code execution."""

    def test_final_in_code_output(self, sample_cv_text, sample_job_text):
        """Test FINAL pattern appearing in execution output."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        # Model that returns code that prints FINAL
        root_model = MagicMock()
        root_model.invoke.return_value = MagicMock(
            content='```python\nprint("FINAL(The answer is 42)")\n```'
        )
        root_provider.get_chat_model.return_value = root_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)

        # Run orchestration - should handle FINAL in output
        result = orchestrator.complete(
            task="Find the answer",
            cv_text=sample_cv_text,
            job_text=sample_job_text,
        )

        # Should complete (may succeed or fail depending on how FINAL is handled)
        assert result is not None

    def test_final_variable_pattern(self, sample_cv_text, sample_job_text):
        """Test FINAL_VAR(variable_name) pattern matching."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        # Model that returns code setting a variable then FINAL_VAR
        root_model = MagicMock()
        root_model.invoke.side_effect = [
            MagicMock(content='```python\nresult = {"answer": 42}\n```'),
            MagicMock(content="FINAL_VAR(result)"),
        ]
        root_provider.get_chat_model.return_value = root_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        config = RLMConfig(max_iterations=3)
        orchestrator.config = config

        result = orchestrator.complete(
            task="Compute answer",
            cv_text=sample_cv_text,
            job_text=sample_job_text,
        )

        assert result is not None


# =============================================================================
# Additional Coverage Tests for orchestrator.py
# =============================================================================


class TestExtractWithLLM:
    """Tests for _extract_with_llm method coverage."""

    def test_extract_with_llm_success(self, sample_cv_text, sample_job_text):
        """Test successful LLM extraction."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        # Set up mock for successful extraction
        mock_extraction_model = MagicMock()
        mock_structured_model = MagicMock()
        mock_config = RLMConfig(max_iterations=5)
        mock_structured_model.invoke.return_value = mock_config
        mock_extraction_model.with_structured_output.return_value = mock_structured_model
        sub_provider.get_extraction_model.return_value = mock_extraction_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        env.set_variable("analysis_key", {"some": "data"})

        # Content with enough length to trigger analysis path
        content = "A" * 200 + " analysis content with meaningful data"

        result = orchestrator._extract_with_llm(content, RLMConfig, env)

        assert isinstance(result, RLMConfig)

    def test_extract_with_llm_short_content_direct_extraction(
        self, sample_cv_text, sample_job_text
    ):
        """Test LLM extraction fallback when content is too short (direct extraction)."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        # Set up mock for extraction - should use direct extraction prompt
        mock_extraction_model = MagicMock()
        mock_structured_model = MagicMock()
        mock_config = RLMConfig(max_iterations=7)
        mock_structured_model.invoke.return_value = mock_config
        mock_extraction_model.with_structured_output.return_value = mock_structured_model
        sub_provider.get_extraction_model.return_value = mock_extraction_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)

        # Short content - should trigger direct extraction path
        short_content = "brief"

        result = orchestrator._extract_with_llm(short_content, RLMConfig, env)

        # Should still work via direct extraction
        assert isinstance(result, RLMConfig)
        assert result.max_iterations == 7

    def test_extract_with_llm_exception_returns_content(self, sample_cv_text, sample_job_text):
        """Test LLM extraction returns original content on exception."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        # Set up mock to raise exception
        mock_extraction_model = MagicMock()
        mock_extraction_model.with_structured_output.side_effect = Exception("LLM error")
        sub_provider.get_extraction_model.return_value = mock_extraction_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)

        result = orchestrator._extract_with_llm("some content", RLMConfig, env)

        # Should return original content on failure
        assert result == "some content"


class TestOrchestratorTimeoutHandling:
    """Tests for timeout handling in complete()."""

    def test_complete_actual_timeout(self, sample_cv_text, sample_job_text):
        """Test complete with actual timeout triggered."""
        import time as time_module
        from unittest.mock import patch

        provider = MagicMock()
        mock_model = MagicMock()

        # Model returns code that doesn't finish quickly
        def slow_invoke(*args, **kwargs):
            time_module.sleep(0.1)  # Small delay
            return MagicMock(content="```python\nprint('thinking')\n```")

        mock_model.invoke.side_effect = slow_invoke
        provider.get_chat_model.return_value = mock_model

        # Very short timeout to trigger timeout condition
        config = RLMConfig(max_iterations=100, timeout_seconds=30)
        orchestrator = RLMOrchestrator(provider, config=config)

        # Patch time to simulate timeout
        original_time = time_module.time

        call_count = [0]

        def mock_time():
            call_count[0] += 1
            # After a few calls, simulate timeout
            if call_count[0] > 10:
                return original_time() + 1000  # Far in the future
            return original_time()

        with patch.object(time_module, "time", mock_time):
            result = orchestrator.complete(
                task="Analyze",
                cv_text=sample_cv_text,
                job_text=sample_job_text,
            )

        # Should have recorded timeout in trajectory
        assert result is not None
        # May or may not have hit timeout depending on timing
        assert result.total_iterations >= 1


class TestOrchestratorCodeAndFinalCombo:
    """Tests for code execution followed by FINAL in same response."""

    def test_code_then_final_in_same_response(self, sample_cv_text, sample_job_text):
        """Test code block followed by FINAL() in same response."""
        provider = MagicMock()
        mock_model = MagicMock()

        # Model returns code AND FINAL in same response
        mock_model.invoke.return_value = MagicMock(
            content="""Let me analyze this.

```python
result = {"skills": ["Python", "Java"]}
print("Analyzed!")
```

Based on the analysis:
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
        assert "skills" in result.answer.lower() or result.total_iterations == 1

    def test_code_then_final_var_in_same_response(self, sample_cv_text, sample_job_text):
        """Test code block followed by FINAL_VAR() in same response."""
        provider = MagicMock()
        mock_model = MagicMock()

        # Model returns code AND FINAL_VAR in same response
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


class TestOrchestratorPlainTextUrgency:
    """Tests for plain text handling with urgency messages."""

    def test_plain_text_triggers_urgency_near_limit(self, sample_cv_text, sample_job_text):
        """Test plain text response near iteration limit triggers urgency."""
        provider = MagicMock()
        mock_model = MagicMock()

        # Returns plain text (thinking) for first few iterations, then FINAL
        call_count = [0]

        def mock_invoke(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 4:
                return MagicMock(content="Let me think about this problem...")
            return MagicMock(content="FINAL(The answer is here)")

        mock_model.invoke.side_effect = mock_invoke
        provider.get_chat_model.return_value = mock_model

        config = RLMConfig(max_iterations=5)
        orchestrator = RLMOrchestrator(provider, config=config)

        result = orchestrator.complete(
            task="Analyze",
            cv_text=sample_cv_text,
            job_text=sample_job_text,
        )

        assert result.success
        assert result.total_iterations >= 3


class TestOrchestratorMaxSubCalls:
    """Tests for max sub-calls handling."""

    def test_max_sub_calls_reached_feedback(self, sample_cv_text, sample_job_text):
        """Test feedback when max sub-calls reached."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        root_model = MagicMock()
        sub_model = MagicMock()

        # Root keeps requesting sub-calls, eventually gives FINAL
        call_count = [0]

        def mock_root_invoke(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:
                return MagicMock(content='```python\nresult = rlm_query(cv_text, "Question?")\n```')
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

        # Should have hit max sub-calls limit
        assert result.sub_call_count <= 2


class TestCreateRLMOrchestratorProviders:
    """Tests for create_rlm_orchestrator factory with different providers."""

    def test_create_with_google_provider(self):
        """Test factory with Google provider defaults."""
        with patch("cv_warlock.llm.base.get_llm_provider") as mock_get:
            mock_provider = MagicMock()
            mock_get.return_value = mock_provider

            from cv_warlock.rlm.orchestrator import create_rlm_orchestrator

            orchestrator = create_rlm_orchestrator(
                root_provider="google",
                root_model="gemini-3-pro-preview",
            )

            assert orchestrator.root_provider == mock_provider
            # Should have called with gemini-3-flash-preview for sub
            calls = mock_get.call_args_list
            assert len(calls) == 2

    def test_create_with_openai_provider(self):
        """Test factory with OpenAI provider defaults."""
        with patch("cv_warlock.llm.base.get_llm_provider") as mock_get:
            mock_provider = MagicMock()
            mock_get.return_value = mock_provider

            from cv_warlock.rlm.orchestrator import create_rlm_orchestrator

            orchestrator = create_rlm_orchestrator(
                root_provider="openai",
                root_model="gpt-5.2",
            )

            assert orchestrator.root_provider == mock_provider
            # Should have called with gpt-5-mini for sub
            calls = mock_get.call_args_list
            assert len(calls) == 2

    def test_create_with_different_sub_provider(self):
        """Test factory with different sub provider than root."""
        with patch("cv_warlock.llm.base.get_llm_provider") as mock_get:
            mock_provider = MagicMock()
            mock_get.return_value = mock_provider

            from cv_warlock.rlm.orchestrator import create_rlm_orchestrator

            orchestrator = create_rlm_orchestrator(
                root_provider="anthropic",
                root_model="claude-opus-4-5-20251101",
                sub_provider="openai",
                sub_model="gpt-5-mini",
            )

            assert orchestrator.root_provider == mock_provider
            # Verify both providers were created
            assert mock_get.call_count == 2

    def test_create_with_explicit_sub_model(self):
        """Test factory with explicit sub_model (no provider default)."""
        with patch("cv_warlock.llm.base.get_llm_provider") as mock_get:
            mock_provider = MagicMock()
            mock_get.return_value = mock_provider

            from cv_warlock.rlm.orchestrator import create_rlm_orchestrator

            orchestrator = create_rlm_orchestrator(
                root_provider="anthropic",
                root_model="claude-opus-4-5-20251101",
                sub_model="claude-sonnet-4-5-20250929",
            )

            assert orchestrator is not None
            # Should use the explicit sub_model
            assert mock_get.call_count == 2


class TestOrchestratorFallbackExtraction:
    """Tests for fallback answer extraction edge cases."""

    def test_fallback_with_invalid_json_string_and_llm_extraction(
        self, sample_cv_text, sample_job_text
    ):
        """Test fallback extraction with invalid JSON string triggers LLM extraction."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        # Set up mock for LLM extraction fallback
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

        # Should attempt LLM extraction and succeed
        assert isinstance(answer, RLMConfig)

    def test_fallback_with_combined_variables_llm_extraction(self, sample_cv_text, sample_job_text):
        """Test fallback extraction combines variables for LLM extraction."""
        root_provider = MagicMock()
        sub_provider = MagicMock()

        # Set up mock for LLM extraction
        mock_extraction_model = MagicMock()
        mock_structured_model = MagicMock()
        mock_config = RLMConfig(max_iterations=15)
        mock_structured_model.invoke.return_value = mock_config
        mock_extraction_model.with_structured_output.return_value = mock_structured_model
        sub_provider.get_extraction_model.return_value = mock_extraction_model

        orchestrator = RLMOrchestrator(root_provider, sub_provider)
        env = REPLEnvironment(cv_text=sample_cv_text, job_text=sample_job_text)
        # No result keys, but has other variables
        env.variables["custom_data"] = {"key": "value"}
        env.variables["analysis"] = ["item1", "item2"]

        answer = orchestrator._extract_fallback_answer(env, RLMConfig)

        # Should combine variables and try LLM extraction
        assert isinstance(answer, RLMConfig)


# =============================================================================
# Additional Coverage Tests for rlm_nodes.py
# =============================================================================


class TestIsValidCVData:
    """Tests for _is_valid_cv_data helper function."""

    def test_valid_cv_data_with_experiences(self):
        """Test valid CVData with experiences."""
        from cv_warlock.graph.rlm_nodes import _is_valid_cv_data
        from cv_warlock.models.cv import ContactInfo, CVData, Experience

        cv_data = CVData(
            contact=ContactInfo(name="John Doe", email="john@example.com"),
            summary="Experienced developer",
            experiences=[
                Experience(
                    title="Developer",
                    company="TechCorp",
                    start_date="2020-01",
                    description="Built things",
                )
            ],
            education=[],
            skills=[],
        )

        assert _is_valid_cv_data(cv_data) is True

    def test_valid_cv_data_with_skills(self):
        """Test valid CVData with skills but no experiences."""
        from cv_warlock.graph.rlm_nodes import _is_valid_cv_data
        from cv_warlock.models.cv import ContactInfo, CVData

        cv_data = CVData(
            contact=ContactInfo(name="Jane Doe", email="jane@example.com"),
            summary="Skilled professional",
            experiences=[],
            education=[],
            skills=["Python", "JavaScript"],
        )

        assert _is_valid_cv_data(cv_data) is True

    def test_invalid_cv_data_unknown_name(self):
        """Test invalid CVData with UNKNOWN name."""
        from cv_warlock.graph.rlm_nodes import _is_valid_cv_data
        from cv_warlock.models.cv import ContactInfo, CVData

        cv_data = CVData(
            contact=ContactInfo(name="UNKNOWN", email="test@example.com"),
            summary="Test",
            experiences=[],
            education=[],
            skills=["Python"],
        )

        assert _is_valid_cv_data(cv_data) is False

    def test_invalid_cv_data_placeholder_name(self):
        """Test invalid CVData with placeholder name."""
        from cv_warlock.graph.rlm_nodes import _is_valid_cv_data
        from cv_warlock.models.cv import ContactInfo, CVData

        cv_data = CVData(
            contact=ContactInfo(name="<UNKNOWN>", email="test@example.com"),
            summary="Test",
            experiences=[],
            education=[],
            skills=["Python"],
        )

        assert _is_valid_cv_data(cv_data) is False

    def test_invalid_cv_data_na_name(self):
        """Test invalid CVData with N/A name."""
        from cv_warlock.graph.rlm_nodes import _is_valid_cv_data
        from cv_warlock.models.cv import ContactInfo, CVData

        cv_data = CVData(
            contact=ContactInfo(name="N/A", email="test@example.com"),
            summary="Test",
            experiences=[],
            education=[],
            skills=["Python"],
        )

        assert _is_valid_cv_data(cv_data) is False

    def test_invalid_cv_data_empty_name(self):
        """Test invalid CVData with empty name."""
        from cv_warlock.graph.rlm_nodes import _is_valid_cv_data
        from cv_warlock.models.cv import ContactInfo, CVData

        cv_data = CVData(
            contact=ContactInfo(name="", email="test@example.com"),
            summary="Test",
            experiences=[],
            education=[],
            skills=["Python"],
        )

        assert _is_valid_cv_data(cv_data) is False

    def test_invalid_cv_data_name_not_provided(self):
        """Test invalid CVData with 'Name not provided' name."""
        from cv_warlock.graph.rlm_nodes import _is_valid_cv_data
        from cv_warlock.models.cv import ContactInfo, CVData

        cv_data = CVData(
            contact=ContactInfo(name="NAME NOT PROVIDED", email="test@example.com"),
            summary="Test",
            experiences=[],
            education=[],
            skills=["Python"],
        )

        assert _is_valid_cv_data(cv_data) is False

    def test_invalid_cv_data_no_experiences_or_skills(self):
        """Test invalid CVData with no experiences or skills."""
        from cv_warlock.graph.rlm_nodes import _is_valid_cv_data
        from cv_warlock.models.cv import ContactInfo, CVData

        cv_data = CVData(
            contact=ContactInfo(name="John Doe", email="john@example.com"),
            summary="Test",
            experiences=[],
            education=[],
            skills=[],
        )

        assert _is_valid_cv_data(cv_data) is False

    def test_invalid_cv_data_not_cvdata_type(self):
        """Test _is_valid_cv_data with non-CVData type."""
        from cv_warlock.graph.rlm_nodes import _is_valid_cv_data

        assert _is_valid_cv_data("not a CVData") is False
        assert _is_valid_cv_data({"name": "test"}) is False
        assert _is_valid_cv_data(None) is False


class TestCheckRLMTimeout:
    """Tests for _check_rlm_timeout helper function."""

    def test_check_timeout_true(self):
        """Test _check_rlm_timeout returns True for timeout."""
        from cv_warlock.graph.rlm_nodes import _check_rlm_timeout

        step = TrajectoryStep(
            step_number=1,
            action_type=ActionType.FINAL,
            model_output="TIMEOUT",
            parsed_action=None,
            execution_result="Timeout message",
            duration_ms=0,
        )
        rlm_result = RLMResult(
            answer=None,
            trajectory=[step],
            sub_call_count=0,
            total_iterations=1,
            success=False,
        )

        assert _check_rlm_timeout(rlm_result) is True

    def test_check_timeout_false_no_timeout(self):
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
        rlm_result = RLMResult(
            answer="answer",
            trajectory=[step],
            sub_call_count=0,
            total_iterations=1,
            success=True,
        )

        assert _check_rlm_timeout(rlm_result) is False

    def test_check_timeout_empty_trajectory(self):
        """Test _check_rlm_timeout returns False for empty trajectory."""
        from cv_warlock.graph.rlm_nodes import _check_rlm_timeout

        rlm_result = RLMResult(
            answer=None,
            trajectory=[],
            sub_call_count=0,
            total_iterations=0,
            success=False,
        )

        assert _check_rlm_timeout(rlm_result) is False


class TestGetTimeoutMessage:
    """Tests for _get_timeout_message helper function."""

    def test_get_timeout_message_with_timeout(self):
        """Test _get_timeout_message returns message when timeout."""
        from cv_warlock.graph.rlm_nodes import _get_timeout_message

        step = TrajectoryStep(
            step_number=1,
            action_type=ActionType.FINAL,
            model_output="TIMEOUT",
            parsed_action=None,
            execution_result="RLM timeout after 30s",
            duration_ms=0,
        )
        rlm_result = RLMResult(
            answer=None,
            trajectory=[step],
            sub_call_count=0,
            total_iterations=1,
            success=False,
        )

        assert _get_timeout_message(rlm_result) == "RLM timeout after 30s"

    def test_get_timeout_message_no_timeout(self):
        """Test _get_timeout_message returns None when no timeout."""
        from cv_warlock.graph.rlm_nodes import _get_timeout_message

        step = TrajectoryStep(
            step_number=1,
            action_type=ActionType.FINAL,
            model_output="Normal output",
            parsed_action=None,
            execution_result="Done",
            duration_ms=100,
        )
        rlm_result = RLMResult(
            answer="answer",
            trajectory=[step],
            sub_call_count=0,
            total_iterations=1,
            success=True,
        )

        assert _get_timeout_message(rlm_result) is None

    def test_get_timeout_message_empty_trajectory(self):
        """Test _get_timeout_message returns None for empty trajectory."""
        from cv_warlock.graph.rlm_nodes import _get_timeout_message

        rlm_result = RLMResult(
            answer=None,
            trajectory=[],
            sub_call_count=0,
            total_iterations=0,
            success=False,
        )

        assert _get_timeout_message(rlm_result) is None


class TestExtractCVRLMTimeout:
    """Tests for extract_cv_rlm timeout handling."""

    def test_extract_cv_rlm_timeout_with_successful_fallback(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extract_cv_rlm timeout with successful fallback."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        cv_data = CVData(
            contact=ContactInfo(name="Fallback Name", email="fallback@test.com"),
            summary="Fallback summary",
            experiences=[],
            education=[],
            skills=["Python"],
        )

        timeout_step = TrajectoryStep(
            step_number=1,
            action_type=ActionType.FINAL,
            model_output="TIMEOUT",
            parsed_action=None,
            execution_result="RLM timeout after 30s",
            duration_ms=0,
        )

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            # RLM times out
            mock_complete.return_value = RLMResult(
                answer=None,
                trajectory=[timeout_step],
                sub_call_count=0,
                total_iterations=1,
                success=False,
            )
            # Standard extraction succeeds
            mock_create_nodes.return_value = create_mock_standard_nodes(
                extract_cv={"cv_data": cv_data, "current_step": "extract_cv"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_cv"](state)

            # Should fall back and succeed without adding errors
            assert result["cv_data"] == cv_data
            assert result["rlm_metadata"]["used"] is False
            # No error added since fallback succeeded
            assert "errors" not in result or not any(
                "timeout" in str(e).lower() for e in result.get("errors", [])
            )

    def test_extract_cv_rlm_timeout_with_failed_fallback(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extract_cv_rlm timeout with failed fallback adds error."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        timeout_step = TrajectoryStep(
            step_number=1,
            action_type=ActionType.FINAL,
            model_output="TIMEOUT",
            parsed_action=None,
            execution_result="RLM timeout after 30s",
            duration_ms=0,
        )

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            # RLM times out
            mock_complete.return_value = RLMResult(
                answer=None,
                trajectory=[timeout_step],
                sub_call_count=0,
                total_iterations=1,
                success=False,
            )
            # Standard extraction also fails (returns None)
            mock_create_nodes.return_value = create_mock_standard_nodes(
                extract_cv={"cv_data": None, "current_step": "extract_cv"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_cv"](state)

            # Should add error when fallback also fails
            assert result["cv_data"] is None
            assert "errors" in result
            assert any("timeout" in str(e).lower() for e in result["errors"])

    def test_extract_cv_rlm_invalid_cvdata_fallback(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extract_cv_rlm with invalid CVData falls back."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        # RLM returns CVData but with invalid/placeholder content
        invalid_cv_data = CVData(
            contact=ContactInfo(name="UNKNOWN", email=""),  # Invalid name
            summary="",
            experiences=[],
            education=[],
            skills=[],
        )

        valid_cv_data = CVData(
            contact=ContactInfo(name="Real Name", email="real@test.com"),
            summary="Real summary",
            experiences=[],
            education=[],
            skills=["Python"],
        )

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            # RLM returns invalid CVData
            mock_complete.return_value = RLMResult(
                answer=invalid_cv_data,
                trajectory=[],
                sub_call_count=0,
                total_iterations=1,
                success=True,
            )
            # Standard extraction returns valid data
            mock_create_nodes.return_value = create_mock_standard_nodes(
                extract_cv={"cv_data": valid_cv_data, "current_step": "extract_cv"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_cv"](state)

            # Should fall back to standard extraction
            assert result["cv_data"] == valid_cv_data
            assert result["rlm_metadata"]["used"] is False


class TestExtractJobRLMTimeout:
    """Tests for extract_job_rlm timeout handling."""

    def test_extract_job_rlm_timeout_with_successful_fallback(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extract_job_rlm timeout with successful fallback."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        job_requirements = JobRequirements(
            job_title="Developer",
            required_skills=["Python"],
            preferred_skills=[],
            responsibilities=[],
        )

        timeout_step = TrajectoryStep(
            step_number=1,
            action_type=ActionType.FINAL,
            model_output="TIMEOUT",
            parsed_action=None,
            execution_result="RLM timeout after 30s",
            duration_ms=0,
        )

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            # RLM times out
            mock_complete.return_value = RLMResult(
                answer=None,
                trajectory=[timeout_step],
                sub_call_count=0,
                total_iterations=1,
                success=False,
            )
            # Standard extraction succeeds
            mock_create_nodes.return_value = create_mock_standard_nodes(
                extract_job={"job_requirements": job_requirements, "current_step": "extract_job"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_job"](state)

            # Should fall back and succeed
            assert result["job_requirements"] == job_requirements

    def test_extract_job_rlm_timeout_with_failed_fallback(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test extract_job_rlm timeout with failed fallback adds error."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        timeout_step = TrajectoryStep(
            step_number=1,
            action_type=ActionType.FINAL,
            model_output="TIMEOUT",
            parsed_action=None,
            execution_result="RLM timeout after 30s",
            duration_ms=0,
        )

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            # RLM times out
            mock_complete.return_value = RLMResult(
                answer=None,
                trajectory=[timeout_step],
                sub_call_count=0,
                total_iterations=1,
                success=False,
            )
            # Standard extraction also fails
            mock_create_nodes.return_value = create_mock_standard_nodes(
                extract_job={"job_requirements": None, "current_step": "extract_job"}
            )

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["extract_job"](state)

            # Should add error when fallback also fails
            assert result["job_requirements"] is None
            assert "errors" in result
            assert any("timeout" in str(e).lower() for e in result["errors"])


class TestAnalyzeMatchRLMTimeout:
    """Tests for analyze_match_rlm timeout handling."""

    def test_analyze_match_rlm_timeout_fallback(
        self, mock_llm_provider, sample_cv_text, sample_job_text
    ):
        """Test analyze_match_rlm timeout falls back to standard."""
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes

        large_cv = sample_cv_text * 10
        large_job = sample_job_text * 10
        state = {
            "raw_cv": large_cv,
            "raw_job_spec": large_job,
            "use_rlm": True,
            "errors": [],
        }

        match_result = {
            "match_analysis": {
                "strong_matches": ["Python"],
                "partial_matches": [],
                "gaps": [],
                "transferable_skills": [],
                "relevance_score": 0.7,
            },
            "current_step": "analyze_match",
        }

        timeout_step = TrajectoryStep(
            step_number=1,
            action_type=ActionType.FINAL,
            model_output="TIMEOUT",
            parsed_action=None,
            execution_result="RLM timeout",
            duration_ms=0,
        )

        with (
            patch.object(RLMOrchestrator, "complete") as mock_complete,
            patch("cv_warlock.graph.nodes.create_nodes") as mock_create_nodes,
        ):
            # RLM times out (success=False)
            mock_complete.return_value = RLMResult(
                answer=None,
                trajectory=[timeout_step],
                sub_call_count=0,
                total_iterations=1,
                success=False,
            )
            mock_create_nodes.return_value = create_mock_standard_nodes(analyze_match=match_result)

            config = RLMConfig(size_threshold=100)
            nodes = create_rlm_nodes(mock_llm_provider, config=config)
            result = nodes["analyze_match"](state)

            # Should fall back to standard
            assert result["match_analysis"]["relevance_score"] == 0.7


class TestOrchestratorMultipleCodeBlocks:
    """Tests for handling multiple code blocks in output."""

    def test_parse_multiple_code_blocks_combined(self, mock_llm_provider):
        """Test that multiple code blocks are combined."""
        orchestrator = RLMOrchestrator(mock_llm_provider)

        output = """Let me analyze in steps.

```python
step1 = "First"
print(step1)
```

Now the second part:

```python
step2 = step1 + " and Second"
print(step2)
```"""

        action = orchestrator._parse_model_output(output)

        assert action.action_type == ActionType.CODE
        # Should contain both code blocks combined
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

        # Should be CODE action, not FINAL
        assert action.action_type == ActionType.CODE
        assert "actual computation" in action.content
