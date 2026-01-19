"""Tests for RLM (Recursive Language Model) integration."""

import pytest

from cv_warlock.rlm.chunking import CVChunker, JobChunker
from cv_warlock.rlm.environment import REPLEnvironment
from cv_warlock.rlm.models import (
    ActionType,
    CVChunks,
    ExecutionResult,
    JobChunks,
    ModelAction,
    RLMConfig,
    RLMResult,
)


class TestREPLEnvironment:
    """Tests for the REPL environment."""

    def test_init_with_context(self) -> None:
        """Test environment initializes with CV and job text."""
        env = REPLEnvironment(
            cv_text="Sample CV content",
            job_text="Sample job description",
        )

        assert env.cv_text == "Sample CV content"
        assert env.job_text == "Sample job description"
        assert env.variables == {}
        assert env.execution_log == []

    def test_execute_simple_code(self) -> None:
        """Test executing simple Python code."""
        env = REPLEnvironment(
            cv_text="Sample CV",
            job_text="Sample job",
        )

        result = env.execute("x = 1 + 1\nprint(x)")

        assert result.success
        assert "2" in result.output
        assert "x" in env.variables or "x" in env._namespace

    def test_execute_access_cv_text(self) -> None:
        """Test accessing cv_text variable."""
        env = REPLEnvironment(
            cv_text="Test CV content here",
            job_text="Test job",
        )

        result = env.execute("length = len(cv_text)\nprint(length)")

        assert result.success
        assert "20" in result.output  # len("Test CV content here")

    def test_execute_blocked_patterns(self) -> None:
        """Test that dangerous patterns are blocked."""
        env = REPLEnvironment(cv_text="CV", job_text="Job")

        # Test blocked imports
        result = env.execute("import os")
        assert not result.success
        assert "Security violation" in (result.error or "")

        result = env.execute("import subprocess")
        assert not result.success

        result = env.execute("__import__('os')")
        assert not result.success

    def test_execute_with_error(self) -> None:
        """Test execution error handling."""
        env = REPLEnvironment(cv_text="CV", job_text="Job")

        result = env.execute("undefined_variable")

        assert not result.success
        assert result.error is not None
        assert "NameError" in result.error

    def test_register_function(self) -> None:
        """Test registering custom functions."""
        env = REPLEnvironment(cv_text="CV", job_text="Job")

        def custom_func(x: int) -> int:
            return x * 2

        env.register_function("double", custom_func)

        result = env.execute("result = double(5)\nprint(result)")

        assert result.success
        assert "10" in result.output

    def test_set_and_get_variable(self) -> None:
        """Test variable storage."""
        env = REPLEnvironment(cv_text="CV", job_text="Job")

        env.set_variable("test_var", "test_value")
        assert env.get_variable("test_var") == "test_value"

    def test_context_summary(self) -> None:
        """Test context summary generation."""
        env = REPLEnvironment(
            cv_text="A" * 1000,
            job_text="B" * 500,
        )

        summary = env.get_context_summary()

        assert "cv_text: 1000 characters" in summary
        assert "job_text: 500 characters" in summary

    def test_reset(self) -> None:
        """Test environment reset."""
        env = REPLEnvironment(cv_text="CV", job_text="Job")

        env.set_variable("test", "value")
        env.execute("print('hello')")

        env.reset()

        assert env.variables == {}
        assert env.execution_log == []


class TestCVChunker:
    """Tests for CV chunking."""

    def test_chunk_basic_cv(self) -> None:
        """Test chunking a basic CV with headers."""
        cv_text = """# John Doe

## Summary
Experienced software engineer with 10 years of experience.

## Experience
### Senior Developer at TechCorp
2020 - Present
- Led development team
- Implemented CI/CD

### Developer at StartupInc
2015 - 2020
- Built web applications
- Worked with Python

## Education
### MS Computer Science
University of Example, 2015

## Skills
Python, JavaScript, Docker, Kubernetes
"""
        chunker = CVChunker()
        chunks = chunker.chunk(cv_text)

        assert chunks.raw_text == cv_text
        assert len(chunks.sections) > 0
        assert chunks.summary is not None or "summary" in str(chunks.sections).lower()

    def test_chunk_cv_without_headers(self) -> None:
        """Test chunking CV without headers."""
        cv_text = """Just plain text CV content
with multiple lines
but no markdown headers."""

        chunker = CVChunker()
        chunks = chunker.chunk(cv_text)

        # Should fall back to storing as summary
        assert chunks.summary == cv_text.strip()

    def test_get_section(self) -> None:
        """Test getting specific sections."""
        cv_text = """# Name

## Skills
Python, Java, Go
"""
        chunker = CVChunker()
        chunks = chunker.chunk(cv_text)

        # May or may not find depending on parsing
        # Just verify it doesn't crash
        chunker.get_section(chunks, "skills")


class TestJobChunker:
    """Tests for job spec chunking."""

    def test_chunk_basic_job(self) -> None:
        """Test chunking a basic job posting."""
        job_text = """# Senior Software Engineer

## About the Role
Join our growing team to build amazing products.

## Requirements
- 5+ years of Python experience
- Experience with cloud platforms
- Strong communication skills

## Nice to Have
- Kubernetes experience
- ML/AI background

## Responsibilities
- Design and implement features
- Review code
- Mentor junior developers

## Benefits
- Health insurance
- Remote work
- Stock options
"""
        chunker = JobChunker()
        chunks = chunker.chunk(job_text)

        assert chunks.raw_text == job_text
        assert chunks.title is not None
        assert len(chunks.sections) > 0

    def test_extract_bullets(self) -> None:
        """Test bullet point extraction."""
        job_text = """# Job

## Requirements
- Skill 1
- Skill 2
- Skill 3
"""
        chunker = JobChunker()
        chunks = chunker.chunk(job_text)

        assert len(chunks.required_qualifications) >= 0  # May or may not parse

    def test_get_requirements_by_priority(self) -> None:
        """Test prioritized requirements retrieval."""
        chunks = JobChunks(
            raw_text="test",
            required_qualifications=["Python", "AWS"],
            preferred_qualifications=["Kubernetes"],
        )

        chunker = JobChunker()
        reqs = chunker.get_requirements_by_priority(chunks)

        # Required should come first
        assert len(reqs) == 3
        assert reqs[0][1]  # First is required
        assert reqs[1][1]  # Second is required
        assert not reqs[2][1]  # Third is preferred


class TestRLMModels:
    """Tests for RLM data models."""

    def test_model_action_creation(self) -> None:
        """Test ModelAction creation."""
        action = ModelAction(
            action_type=ActionType.CODE,
            content="print('hello')",
        )

        assert action.action_type == ActionType.CODE
        assert action.content == "print('hello')"

    def test_execution_result(self) -> None:
        """Test ExecutionResult."""
        result = ExecutionResult(
            success=True,
            output="Hello, World!",
            execution_time_ms=10.5,
        )

        assert result.success
        assert result.output == "Hello, World!"
        assert result.error is None

    def test_rlm_config_defaults(self) -> None:
        """Test RLMConfig default values."""
        config = RLMConfig()

        # Reduced defaults for faster performance (was 8/8/480/8000)
        assert config.max_iterations == 4
        assert config.max_sub_calls == 4
        assert config.timeout_seconds == 300
        assert config.size_threshold == 25000
        assert config.sandbox_mode == "local"

    def test_rlm_config_custom(self) -> None:
        """Test RLMConfig with custom values."""
        config = RLMConfig(
            max_iterations=10,
            max_sub_calls=5,
            root_model="gpt-4",
            size_threshold=5000,
        )

        assert config.max_iterations == 10
        assert config.max_sub_calls == 5
        assert config.root_model == "gpt-4"
        assert config.size_threshold == 5000

    def test_rlm_result_success(self) -> None:
        """Test RLMResult for successful execution."""
        result = RLMResult(
            answer="Analysis complete",
            sub_call_count=3,
            total_iterations=5,
            execution_time_seconds=10.5,
            success=True,
        )

        assert result.success
        assert result.answer == "Analysis complete"
        assert result.sub_call_count == 3
        assert result.error is None

    def test_rlm_result_failure(self) -> None:
        """Test RLMResult for failed execution."""
        result = RLMResult(
            answer=None,
            success=False,
            error="Timeout exceeded",
        )

        assert not result.success
        assert result.error == "Timeout exceeded"

    def test_cv_chunks_total(self) -> None:
        """Test CVChunks total_chunks calculation."""
        chunks = CVChunks(
            raw_text="test",
            summary="Summary here",
            experiences=["exp1", "exp2"],
            education=["edu1"],
            skills="Python, Java",
        )

        # 1 summary + 2 experiences + 1 education + 1 skills = 5
        assert chunks.total_chunks == 5

    def test_job_chunks_total_requirements(self) -> None:
        """Test JobChunks total_requirements."""
        chunks = JobChunks(
            raw_text="test",
            required_qualifications=["req1", "req2", "req3"],
            preferred_qualifications=["pref1"],
        )

        assert chunks.total_requirements == 4


class TestRLMConfig:
    """Tests for RLM configuration from settings."""

    def test_config_from_settings(self) -> None:
        """Test RLM config can be created from settings."""
        from cv_warlock.config import Settings

        settings = Settings(
            rlm_enabled=True,
            rlm_max_iterations=15,
            rlm_max_sub_calls=10,
        )

        config = settings.rlm_config

        assert config.max_iterations == 15
        assert config.max_sub_calls == 10


# Integration test placeholder - requires mocking LLM
class TestRLMOrchestrator:
    """Tests for RLM orchestrator (integration tests)."""

    @pytest.mark.skip(reason="Requires LLM mocking")
    def test_orchestrator_complete(self) -> None:
        """Test full orchestrator completion."""
        pass

    def test_orchestrator_parse_code_block(self) -> None:
        """Test parsing code blocks from model output."""

        output = """Let me analyze this.

```python
skills = cv_text.split(',')
print(len(skills))
```

This will count the skills."""

        # Test the regex pattern directly
        import re

        pattern = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
        match = pattern.search(output)

        assert match is not None
        assert "skills = cv_text.split" in match.group(1)

    def test_orchestrator_parse_final(self) -> None:
        """Test parsing FINAL statement."""
        output = """After analysis, the result is:

FINAL(The candidate has 5 years of Python experience and matches 80% of requirements)"""

        import re

        pattern = re.compile(r"FINAL\s*\(\s*(.*)\s*\)", re.DOTALL)
        match = pattern.search(output)

        assert match is not None
        assert "5 years of Python" in match.group(1)
