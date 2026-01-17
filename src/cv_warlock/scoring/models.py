"""Pydantic models for hybrid ATS-style scoring."""

from typing import TypedDict

from pydantic import BaseModel, Field


class ScoreBreakdown(TypedDict):
    """Detailed breakdown of algorithmic sub-scores (0-1 scale)."""

    exact_skill_match: float  # Exact string match for skills
    semantic_skill_match: float  # Embedding-based skill similarity
    document_similarity: float  # CV-job document embedding similarity
    experience_years_fit: float  # Years of experience vs requirement
    education_match: float  # Education level match
    recency_score: float  # Recent experience relevance


class AlgorithmicScores(BaseModel):
    """Computed algorithmic scores with knockout status."""

    # Individual sub-scores (0-1)
    exact_skill_match: float = Field(ge=0, le=1)
    semantic_skill_match: float = Field(ge=0, le=1)
    document_similarity: float = Field(ge=0, le=1)
    experience_years_fit: float = Field(ge=0, le=1)
    education_match: float = Field(ge=0, le=1)
    recency_score: float = Field(ge=0, le=1)

    # Combined algorithmic score
    total: float = Field(ge=0, le=1)

    # Knockout rule status
    knockout_triggered: bool = False
    knockout_reason: str | None = None

    def to_breakdown(self) -> ScoreBreakdown:
        """Convert to TypedDict for state storage."""
        return ScoreBreakdown(
            exact_skill_match=self.exact_skill_match,
            semantic_skill_match=self.semantic_skill_match,
            document_similarity=self.document_similarity,
            experience_years_fit=self.experience_years_fit,
            education_match=self.education_match,
            recency_score=self.recency_score,
        )


class LLMAssessmentOutput(BaseModel):
    """Structured output from LLM qualitative assessment."""

    transferable_skills: list[str] = Field(
        default_factory=list,
        description="Non-obvious transferable skills that algorithms miss",
    )
    contextual_strengths: list[str] = Field(
        default_factory=list,
        description="Narrative strengths for this specific role",
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Qualitative concerns (empty if none)",
    )
    adjustment: float = Field(
        default=0.0,
        ge=-0.1,
        le=0.1,
        description="Score adjustment recommendation (-0.1 to +0.1)",
    )
    adjustment_rationale: str = Field(
        default="",
        description="Brief explanation for adjustment",
    )


class HybridMatchResult(TypedDict):
    """Final hybrid match result, compatible with MatchAnalysis structure."""

    # Core fields (same as MatchAnalysis for compatibility)
    strong_matches: list[str]
    partial_matches: list[str]
    gaps: list[str]
    transferable_skills: list[str]
    relevance_score: float  # Final combined score

    # Hybrid scoring metadata
    score_breakdown: ScoreBreakdown
    algorithmic_score: float  # Pre-LLM algorithmic score
    llm_adjustment: float  # LLM adjustment applied
    knockout_triggered: bool
    knockout_reason: str | None
    scoring_method: str  # "hybrid" or "llm_only"
