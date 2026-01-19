"""Pydantic models for RLM components."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Type of action the model wants to perform."""

    CODE = "code"  # Execute Python code
    QUERY = "query"  # Sub-LLM query on a chunk
    FINAL = "final"  # Final answer ready


class ModelAction(BaseModel):
    """Parsed action from model output."""

    action_type: ActionType
    content: str  # Code to execute, query text, or final answer
    context_var: str | None = None  # Variable name for query context
    question: str | None = None  # Question for sub-query


class ExecutionEntry(BaseModel):
    """Log entry for a single execution step."""

    step_number: int
    action_type: ActionType
    input_content: str  # Code or query
    output: str | None  # Result or error
    success: bool
    execution_time_ms: float
    tokens_used: int | None = None


class TrajectoryStep(BaseModel):
    """A step in the RLM trajectory for observability."""

    step_number: int
    action_type: ActionType
    model_output: str  # Raw model output
    parsed_action: ModelAction | None
    execution_result: str | None
    sub_call_made: bool = False
    sub_call_context: str | None = None
    sub_call_question: str | None = None
    sub_call_answer: str | None = None
    tokens_used: int | None = None
    duration_ms: float


class RLMResult(BaseModel):
    """Result from an RLM orchestration run."""

    answer: Any  # Final answer (can be structured or text)
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    sub_call_count: int = 0
    total_iterations: int = 0
    total_tokens: int = 0
    execution_time_seconds: float = 0.0
    success: bool = True
    error: str | None = None
    intermediate_findings: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Result from executing code in the REPL environment."""

    success: bool
    output: str  # stdout/return value
    error: str | None = None
    variables_modified: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0


class SubCallResult(BaseModel):
    """Result from a sub-LLM call."""

    question: str
    context_preview: str  # First N chars of context
    answer: str
    tokens_used: int | None = None
    duration_ms: float


class CVChunks(BaseModel):
    """Parsed CV structure with chunks."""

    raw_text: str
    sections: dict[str, str] = Field(default_factory=dict)
    contact: str | None = None
    summary: str | None = None
    experiences: list[str] = Field(default_factory=list)  # Individual job chunks
    education: list[str] = Field(default_factory=list)
    skills: str | None = None
    projects: list[str] = Field(default_factory=list)
    certifications: str | None = None
    other_sections: dict[str, str] = Field(default_factory=dict)

    @property
    def total_chunks(self) -> int:
        """Total number of distinct chunks."""
        return (
            len(self.experiences)
            + len(self.education)
            + len(self.projects)
            + (1 if self.summary else 0)
            + (1 if self.skills else 0)
            + (1 if self.certifications else 0)
            + len(self.other_sections)
        )


class JobChunks(BaseModel):
    """Parsed job spec structure with chunks."""

    raw_text: str
    sections: dict[str, str] = Field(default_factory=dict)
    title: str | None = None
    company: str | None = None
    overview: str | None = None
    required_qualifications: list[str] = Field(default_factory=list)
    preferred_qualifications: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    benefits: str | None = None
    company_info: str | None = None
    other_sections: dict[str, str] = Field(default_factory=dict)

    @property
    def total_requirements(self) -> int:
        """Total number of requirements to analyze."""
        return len(self.required_qualifications) + len(self.preferred_qualifications)


class RLMConfig(BaseModel):
    """Configuration for RLM orchestration."""

    # Model configuration
    root_provider: Literal["openai", "anthropic", "google"] = "anthropic"
    root_model: str = "claude-opus-4-5-20251101"
    sub_provider: Literal["openai", "anthropic", "google"] | None = None
    sub_model: str | None = None  # Defaults to a faster model

    # Execution limits (lower = faster, less thorough)
    max_iterations: int = Field(default=8, ge=1, le=100)
    max_sub_calls: int = Field(default=8, ge=1, le=50)
    timeout_seconds: int = Field(default=480, ge=30, le=600)

    # Context thresholds
    size_threshold: int = Field(
        default=8000,
        description="Character count to trigger RLM mode",
    )

    # Sandbox configuration
    sandbox_mode: Literal["local", "docker", "modal"] = "local"

    # Output preferences
    verbose_trajectory: bool = False  # Include full model outputs in trajectory
