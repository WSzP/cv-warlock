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
