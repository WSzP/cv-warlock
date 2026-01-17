"""LangGraph state models."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from cv_warlock.models.cv import CVData
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.models.reasoning import (
    ExperienceGenerationResult,
    GenerationContext,
    SkillsGenerationResult,
    SummaryGenerationResult,
)


class ScoreBreakdown(TypedDict):
    """Detailed breakdown of algorithmic sub-scores (0-1 scale).

    Used by hybrid scoring to provide explainable, reproducible scores.
    """

    exact_skill_match: float  # Exact string match for skills
    semantic_skill_match: float  # Embedding-based skill similarity
    document_similarity: float  # CV-job document embedding similarity
    experience_years_fit: float  # Years of experience vs requirement
    education_match: float  # Education level match
    recency_score: float  # Recent experience relevance


class MatchAnalysis(TypedDict):
    """Analysis of how CV matches job requirements.

    This base type is used by the LLM-only scorer. The hybrid scorer
    returns HybridMatchResult which extends this with score_breakdown.
    """

    strong_matches: list[str]  # Skills/experience that match well
    partial_matches: list[str]  # Skills that partially match
    gaps: list[str]  # Missing requirements
    transferable_skills: list[str]  # Skills that can be framed as relevant
    relevance_score: float  # 0-1 score


class TailoringPlan(TypedDict):
    """Plan for how to tailor the CV."""

    summary_focus: list[str]  # Key points for summary
    experiences_to_emphasize: list[str]  # Which experiences to highlight
    skills_to_highlight: list[str]  # Priority skills
    achievements_to_feature: list[str]  # Key achievements
    keywords_to_incorporate: list[str]  # ATS keywords
    sections_to_reorder: list[str]  # Section ordering


class StepTiming(TypedDict):
    """Timing information for a single workflow step."""

    step_name: str
    start_time: float  # Unix timestamp
    end_time: float | None  # Unix timestamp, None if in progress
    duration_seconds: float | None  # Computed duration


class CVWarlockState(TypedDict):
    """Main state object for the CV tailoring workflow.

    When use_cot=True, generation is slower (3-4x more LLM calls) but produces
    significantly higher quality tailored CVs through chain-of-thought reasoning.
    """

    # Input data
    raw_cv: str  # Original CV text
    raw_job_spec: str  # Original job specification text

    # Settings
    assume_all_tech_skills: bool  # If True, assume user has all tech skills from job spec
    use_cot: bool  # If True, use chain-of-thought reasoning (slower but better quality)
    lookback_years: int | None  # Override for lookback window (None = use settings default)

    # Extracted structured data
    cv_data: CVData | None
    job_requirements: JobRequirements | None

    # Analysis results
    match_analysis: MatchAnalysis | None
    tailoring_plan: TailoringPlan | None

    # Tailored sections
    tailored_summary: str | None
    tailored_experiences: list[str] | None
    tailored_skills: list[str] | None

    # Output
    tailored_cv: str | None  # Final tailored CV in markdown

    # Chain-of-thought reasoning outputs (for debugging/transparency)
    summary_reasoning_result: SummaryGenerationResult | None
    experience_reasoning_results: list[ExperienceGenerationResult] | None
    skills_reasoning_result: SkillsGenerationResult | None
    generation_context: GenerationContext | None

    # Quality metrics
    total_refinement_iterations: int  # How many refinement loops were needed
    quality_scores: dict[str, str] | None  # Section -> QualityLevel mapping

    # Timing information
    step_timings: list[StepTiming]  # Timing for each completed step
    current_step_start: float | None  # Start time of current step (Unix timestamp)
    total_generation_time: float | None  # Total time in seconds when complete

    # Workflow tracking
    messages: Annotated[list[BaseMessage], add_messages]
    current_step: str
    current_step_description: str  # Human-readable description for UI
    errors: list[str]
