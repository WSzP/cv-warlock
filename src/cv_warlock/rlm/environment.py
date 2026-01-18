"""REPL Environment for RLM execution.

Provides a sandboxed Python environment where the CV and job spec
are stored as variables, and the model can execute code to explore them.
"""

import io
import re
import time
import traceback
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Literal

from cv_warlock.rlm.models import CVChunks, ExecutionResult, JobChunks


class REPLEnvironment:
    """Manages the execution environment for RLM.

    The CV and job spec are stored as variables (not in prompt)
    and the model can execute code to query/manipulate them.

    Security: In production, use sandbox_mode="docker" for isolation.
    Local mode is for development/testing only.
    """

    # Restricted builtins for safety
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "bool",
        "chr",
        "dict",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "int",
        "isinstance",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "ord",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
    }

    # Blocked patterns in code for security
    BLOCKED_PATTERNS = [
        r"import\s+os",
        r"import\s+sys",
        r"import\s+subprocess",
        r"from\s+os\s+import",
        r"from\s+sys\s+import",
        r"__import__",
        r"exec\s*\(",
        r"eval\s*\(",
        r"open\s*\(",
        r"file\s*\(",
        r"compile\s*\(",
        r"globals\s*\(",
        r"locals\s*\(",
        r"getattr\s*\(",
        r"setattr\s*\(",
        r"delattr\s*\(",
        r"__builtins__",
        r"__class__",
        r"__bases__",
        r"__subclasses__",
    ]

    def __init__(
        self,
        cv_text: str,
        job_text: str,
        cv_chunks: CVChunks | None = None,
        job_chunks: JobChunks | None = None,
        sandbox_mode: Literal["local", "docker", "modal"] = "local",
        max_output_length: int = 5000,
        max_execution_time: float = 10.0,
    ):
        """Initialize the REPL environment.

        Args:
            cv_text: Full CV text to analyze.
            job_text: Full job specification text.
            cv_chunks: Pre-parsed CV chunks (optional).
            job_chunks: Pre-parsed job chunks (optional).
            sandbox_mode: Execution isolation mode.
            max_output_length: Maximum chars to return from execution.
            max_execution_time: Maximum seconds for code execution.
        """
        self.cv_text = cv_text
        self.job_text = job_text
        self.cv_chunks = cv_chunks
        self.job_chunks = job_chunks
        self.sandbox_mode = sandbox_mode
        self.max_output_length = max_output_length
        self.max_execution_time = max_execution_time

        # User-defined variables (intermediate results)
        self.variables: dict[str, Any] = {}

        # Registered helper functions
        self._functions: dict[str, Callable[..., Any]] = {}

        # Execution history
        self.execution_log: list[ExecutionResult] = []

        # Initialize the namespace with context
        self._namespace = self._create_namespace()

    def _create_namespace(self) -> dict[str, Any]:
        """Create the execution namespace with context and helpers."""
        # Start with safe builtins
        safe_builtins = {
            name: getattr(__builtins__, name) if hasattr(__builtins__, name) else None
            for name in self.SAFE_BUILTINS
        }
        # Handle case where __builtins__ is a dict (in exec context)
        if isinstance(__builtins__, dict):
            safe_builtins = {name: __builtins__.get(name) for name in self.SAFE_BUILTINS}

        namespace = {
            "__builtins__": safe_builtins,
            # Context variables (read-only from model perspective)
            "cv_text": self.cv_text,
            "job_text": self.job_text,
            "cv_sections": self.cv_chunks.sections if self.cv_chunks else {},
            "job_sections": self.job_chunks.sections if self.job_chunks else {},
            # Mutable results storage
            "results": self.variables,
            # Allow re import for regex
            "re": re,
        }

        # Add structured chunks if available
        if self.cv_chunks:
            namespace["cv_experiences"] = self.cv_chunks.experiences
            namespace["cv_education"] = self.cv_chunks.education
            namespace["cv_skills"] = self.cv_chunks.skills
            namespace["cv_summary"] = self.cv_chunks.summary

        if self.job_chunks:
            namespace["job_requirements"] = self.job_chunks.required_qualifications
            namespace["job_preferred"] = self.job_chunks.preferred_qualifications
            namespace["job_responsibilities"] = self.job_chunks.responsibilities

        # Add registered functions
        namespace.update(self._functions)

        return namespace

    def register_function(self, name: str, func: Callable[..., Any]) -> None:
        """Register a helper function accessible from code.

        Args:
            name: Function name in the namespace.
            func: The callable to register.
        """
        self._functions[name] = func
        self._namespace[name] = func

    def _validate_code(self, code: str) -> tuple[bool, str | None]:
        """Validate code for security issues.

        Returns:
            Tuple of (is_safe, error_message).
        """
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Blocked pattern detected: {pattern}"

        return True, None

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in the sandboxed environment.

        Args:
            code: Python code to execute.

        Returns:
            ExecutionResult with output and status.
        """
        start_time = time.time()

        # Validate code safety
        is_safe, error = self._validate_code(code)
        if not is_safe:
            result = ExecutionResult(
                success=False,
                output="",
                error=f"Security violation: {error}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            self.execution_log.append(result)
            return result

        if self.sandbox_mode == "local":
            return self._execute_local(code, start_time)
        elif self.sandbox_mode == "docker":
            return self._execute_docker(code, start_time)
        elif self.sandbox_mode == "modal":
            return self._execute_modal(code, start_time)
        else:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Unknown sandbox mode: {self.sandbox_mode}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _execute_local(self, code: str, start_time: float) -> ExecutionResult:
        """Execute code locally (development only)."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Track which variables existed before execution
        vars_before = set(self._namespace.keys())

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, self._namespace)

            # Capture output
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            output = stdout_output
            if stderr_output:
                output += f"\nSTDERR: {stderr_output}"

            # Truncate if too long
            if len(output) > self.max_output_length:
                output = output[: self.max_output_length] + "\n... (truncated)"

            # Track new variables
            vars_after = set(self._namespace.keys())
            new_vars = list(vars_after - vars_before)

            # Copy new variables to results storage
            for var in new_vars:
                if not var.startswith("_"):
                    self.variables[var] = self._namespace[var]

            result = ExecutionResult(
                success=True,
                output=output or "(no output)",
                variables_modified=new_vars,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            tb = traceback.format_exc()
            result = ExecutionResult(
                success=False,
                output="",
                error=f"{type(e).__name__}: {e}\n{tb}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        self.execution_log.append(result)
        return result

    def _execute_docker(self, code: str, start_time: float) -> ExecutionResult:
        """Execute code in Docker container (production).

        TODO: Implement Docker-based sandboxing.
        """
        # For now, fall back to local with a warning
        import logging

        logging.warning("Docker sandbox not yet implemented, falling back to local execution")
        return self._execute_local(code, start_time)

    def _execute_modal(self, code: str, start_time: float) -> ExecutionResult:
        """Execute code in Modal sandbox (scalable production).

        TODO: Implement Modal-based sandboxing.
        """
        import logging

        logging.warning("Modal sandbox not yet implemented, falling back to local execution")
        return self._execute_local(code, start_time)

    def get_variable(self, name: str) -> Any:
        """Retrieve a stored variable.

        Args:
            name: Variable name.

        Returns:
            Variable value or None if not found.
        """
        return self.variables.get(name) or self._namespace.get(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Store an intermediate result.

        Args:
            name: Variable name.
            value: Value to store.
        """
        self.variables[name] = value
        self._namespace[name] = value

    def get_context_summary(self) -> str:
        """Get a summary of available context for prompting.

        Returns:
            String describing available variables and their sizes.
        """
        lines = [
            "## Available Context:",
            f"- cv_text: {len(self.cv_text)} characters",
            f"- job_text: {len(self.job_text)} characters",
        ]

        if self.cv_chunks:
            lines.extend(
                [
                    f"- cv_sections: {len(self.cv_chunks.sections)} sections",
                    f"- cv_experiences: {len(self.cv_chunks.experiences)} items",
                    f"- cv_education: {len(self.cv_chunks.education)} items",
                ]
            )

        if self.job_chunks:
            lines.extend(
                [
                    f"- job_sections: {len(self.job_chunks.sections)} sections",
                    f"- job_requirements: {len(self.job_chunks.required_qualifications)} items",
                    f"- job_preferred: {len(self.job_chunks.preferred_qualifications)} items",
                ]
            )

        if self.variables:
            lines.append("\n## Stored Results:")
            for name, value in self.variables.items():
                value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                lines.append(f"- {name}: {value_str}")

        if self._functions:
            lines.append("\n## Available Functions:")
            for name in self._functions:
                lines.append(f"- {name}()")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset the environment to initial state."""
        self.variables.clear()
        self.execution_log.clear()
        self._namespace = self._create_namespace()
