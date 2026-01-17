"""LangGraph state models."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from cv_warlock.models.cv import CVData
from cv_warlock.models.job_spec import JobRequirements


class MatchAnalysis(TypedDict):
    """Analysis of how CV matches job requirements."""

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


class CVWarlockState(TypedDict):
    """Main state object for the CV tailoring workflow."""

    # Input data
    raw_cv: str  # Original CV text
    raw_job_spec: str  # Original job specification text

    # Settings
    assume_all_tech_skills: bool  # If True, assume user has all tech skills from job spec

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

    # Workflow tracking
    messages: Annotated[list[BaseMessage], add_messages]
    current_step: str
    errors: list[str]
