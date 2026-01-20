"""Chain-of-thought reasoning models for CV generation.

These models capture intermediate reasoning steps, quality critiques,
and generation results for each CV section. This enables:
- Transparent reasoning that can be audited/debugged
- Self-critique and refinement loops
- Cross-section context for coherent output
"""

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _parse_stringified_list(value: Any) -> list[str]:
    """Parse a value that may be a stringified JSON list.

    Some smaller LLMs (e.g., Haiku) return list fields as JSON strings
    instead of actual lists. This function handles both cases.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
        # If it's a non-list string, return as single-item list
        return [value] if value else []
    return []


class QualityLevel(str, Enum):
    """Quality assessment levels for generated content."""

    EXCELLENT = "excellent"
    GOOD = "good"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


# =============================================================================
# SUMMARY REASONING MODELS
# =============================================================================


class SummaryReasoning(BaseModel):
    """Intermediate reasoning for professional summary generation.

    Captures the strategic thinking before writing the summary.
    """

    # Analysis phase
    target_title_match: str = Field(
        description="How the candidate's title should be positioned relative to target role"
    )
    key_keywords_to_include: list[str] = Field(
        description="Exact keywords from job posting to incorporate (max 5)"
    )
    strongest_metric: str = Field(
        description="The single most impressive relevant number/achievement to highlight"
    )
    unique_differentiator: str = Field(
        description="What makes this candidate stand out for THIS specific role"
    )

    # Strategy phase
    hook_strategy: str = Field(
        description="The opening hook formula: [Title] + [years] + [domain] + [differentiator]"
    )
    value_proposition: str = Field(description="The core value statement with metric")
    fit_statement: str = Field(description="How to connect to specific job requirements")

    # Risk assessment
    aspects_to_avoid: list[str] = Field(
        default_factory=list,
        description="Topics or phrasings to avoid (gaps, irrelevant experience)",
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence in the reasoning quality (0-1)"
    )

    # Validators for list fields (handles Haiku returning stringified lists)
    @field_validator("key_keywords_to_include", "aspects_to_avoid", mode="before")
    @classmethod
    def parse_list_fields(cls, v: Any) -> list[str]:
        return _parse_stringified_list(v)


class SummaryCritique(BaseModel):
    """Self-critique for generated professional summary.

    Evaluates the summary against proven quality criteria.
    """

    # Quality checks (based on recruiter research)
    has_strong_opening_hook: bool = Field(
        description="Does the first sentence immediately establish relevant identity?"
    )
    includes_quantified_achievement: bool = Field(
        description="Is there at least one hard number/metric?"
    )
    mirrors_job_keywords: bool = Field(
        description="Are 2-3 exact job posting terms naturally incorporated?"
    )
    appropriate_length: bool = Field(description="Is it 2-4 sentences, not longer?")
    avoids_fluff: bool = Field(
        description="Free of weak adjectives like 'passionate', 'driven', 'motivated'?"
    )

    # Overall assessment
    quality_level: QualityLevel = Field(description="Overall quality assessment")
    issues_found: list[str] = Field(
        default_factory=list, description="Specific issues that need fixing"
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list, description="Actionable suggestions for refinement"
    )
    should_refine: bool = Field(description="Does this need another iteration?")

    @field_validator("issues_found", "improvement_suggestions", mode="before")
    @classmethod
    def parse_list_fields(cls, v: Any) -> list[str]:
        return _parse_stringified_list(v)


class SummaryGenerationResult(BaseModel):
    """Complete result of summary generation with full reasoning chain.

    Contains reasoning, initial generation, critique, and final output.
    """

    reasoning: SummaryReasoning
    generated_summary: str = Field(description="Initial generated summary")
    critique: SummaryCritique
    refinement_count: int = Field(default=0, description="Number of refinement iterations")
    final_summary: str = Field(description="The final summary after any refinements")


# =============================================================================
# EXPERIENCE REASONING MODELS
# =============================================================================


class BulletReasoning(BaseModel):
    """Reasoning for transforming a single experience bullet point."""

    original_content: str = Field(description="What the original achievement/duty said")
    relevance_to_job: str = Field(description="How this connects to target role requirements")
    metric_identified: str | None = Field(
        default=None, description="Number/metric to highlight (if available)"
    )
    power_verb_choice: str = Field(description="Action verb to lead with")
    keyword_injection: list[str] = Field(
        default_factory=list, description="Job keywords to incorporate naturally"
    )
    reframed_bullet: str = Field(description="The transformed bullet point")

    @field_validator("keyword_injection", mode="before")
    @classmethod
    def parse_list_fields(cls, v: Any) -> list[str]:
        return _parse_stringified_list(v)


class ExperienceReasoning(BaseModel):
    """Intermediate reasoning for experience entry tailoring.

    Determines how to present each job based on relevance to target role.
    """

    relevance_score: float = Field(
        ge=0.0, le=1.0, description="How relevant is this experience to target role (0-1)"
    )
    emphasis_strategy: str = Field(
        description="HIGH: Full detail (4-5 bullets), MED: Moderate (3-4), LOW: Brief (2-3)"
    )
    transferable_skills_identified: list[str] = Field(
        default_factory=list, description="Skills from this role that transfer to target"
    )
    achievements_to_prioritize: list[str] = Field(
        default_factory=list, description="Which achievements best demonstrate fit"
    )
    keywords_to_incorporate: list[str] = Field(
        default_factory=list, description="Exact job posting terms to weave in"
    )
    bullet_reasoning: list[BulletReasoning] = Field(
        default_factory=list, description="Detailed reasoning for each bullet point"
    )
    aspects_to_downplay: list[str] = Field(
        default_factory=list, description="What to minimize or omit from this role"
    )

    @field_validator(
        "transferable_skills_identified",
        "achievements_to_prioritize",
        "keywords_to_incorporate",
        "aspects_to_downplay",
        mode="before",
    )
    @classmethod
    def parse_list_fields(cls, v: Any) -> list[str]:
        return _parse_stringified_list(v)


class ExperienceCritique(BaseModel):
    """Self-critique for tailored experience bullets.

    Evaluates against impact-first principles.
    """

    # Per-bullet quality checks
    all_bullets_start_with_power_verb: bool = Field(
        description="Every bullet starts with strong action verb (past tense)?"
    )
    all_bullets_show_impact: bool = Field(
        description="Every bullet demonstrates clear result/impact?"
    )
    metrics_present_where_possible: bool = Field(
        description="Metrics included for achievements that had them?"
    )
    relevant_keywords_incorporated: bool = Field(
        description="Job posting keywords naturally woven in?"
    )
    bullets_appropriately_ordered: bool = Field(
        description="Most relevant/impactful bullets listed first?"
    )

    # Overall assessment
    quality_level: QualityLevel
    weak_bullets: list[str] = Field(
        default_factory=list, description="Bullets that need improvement with reasons"
    )
    improvement_suggestions: list[str] = Field(default_factory=list)
    should_refine: bool

    @field_validator("weak_bullets", "improvement_suggestions", mode="before")
    @classmethod
    def parse_list_fields(cls, v: Any) -> list[str]:
        return _parse_stringified_list(v)


class ExperienceGenerationResult(BaseModel):
    """Complete result for one experience entry's tailoring."""

    experience_title: str
    experience_company: str
    reasoning: ExperienceReasoning
    generated_bullets: list[str] = Field(description="Initial generated bullets")
    critique: ExperienceCritique
    refinement_count: int = Field(default=0)
    final_bullets: list[str] = Field(description="Final bullets after refinements")

    @field_validator("generated_bullets", "final_bullets", mode="before")
    @classmethod
    def parse_list_fields(cls, v: Any) -> list[str]:
        return _parse_stringified_list(v)


class SingleExperienceReasoning(BaseModel):
    """Reasoning for a single experience within a batch."""

    experience_index: int = Field(description="Index of the experience (0-based)")
    title: str = Field(description="Job title for this experience")
    company: str = Field(description="Company name for this experience")
    reasoning: ExperienceReasoning = Field(description="Full reasoning for this experience")


class BatchExperienceReasoning(BaseModel):
    """Batch reasoning result for multiple experiences in a single LLM call.

    This model enables processing all experiences in one API call instead of N calls,
    reducing latency by ~50% for the reasoning phase.
    """

    experiences: list[SingleExperienceReasoning] = Field(
        description="Reasoning for each experience, indexed by position"
    )


# =============================================================================
# SKILLS REASONING MODELS
# =============================================================================


class SkillsReasoning(BaseModel):
    """Intermediate reasoning for ATS-optimized skills section.

    Plans terminology, categorization, and ordering for maximum ATS score.
    """

    # Matching analysis
    required_skills_matched: list[str] = Field(
        default_factory=list, description="Required skills candidate has (exact match)"
    )
    required_skills_missing: list[str] = Field(
        default_factory=list, description="Required skills candidate lacks"
    )
    preferred_skills_matched: list[str] = Field(
        default_factory=list, description="Preferred skills candidate has"
    )

    # Terminology decisions
    terminology_mapping: dict[str, str] = Field(
        default_factory=dict, description="Candidate term -> Job posting exact term"
    )
    dual_format_terms: list[str] = Field(
        default_factory=list,
        description="Terms to include both ways: 'AWS (Amazon Web Services)'",
    )

    # Organization strategy
    category_groupings: dict[str, list[str]] = Field(
        default_factory=dict, description="How to group skills by category"
    )
    ordering_rationale: str = Field(default="", description="Why skills are ordered this way")

    # Omissions
    skills_to_omit: list[str] = Field(
        default_factory=list, description="Candidate skills irrelevant to this role"
    )

    @field_validator(
        "required_skills_matched",
        "required_skills_missing",
        "preferred_skills_matched",
        "dual_format_terms",
        "skills_to_omit",
        mode="before",
    )
    @classmethod
    def parse_list_fields(cls, v: Any) -> list[str]:
        return _parse_stringified_list(v)


class SkillsCritique(BaseModel):
    """Self-critique for skills section ATS optimization."""

    # ATS optimization checks
    all_required_skills_present: bool = Field(
        description="All required skills (that candidate has) included?"
    )
    uses_exact_job_terminology: bool = Field(description="Uses exact wording from job posting?")
    appropriate_categorization: bool = Field(
        description="Skills grouped logically matching job structure?"
    )
    no_irrelevant_skills: bool = Field(description="No skills that dilute relevance signal?")
    no_fabricated_skills: bool = Field(description="No skills candidate doesn't actually have?")

    # Quality assessment
    quality_level: QualityLevel
    missing_critical_terms: list[str] = Field(
        default_factory=list, description="Important terms that should be added"
    )
    improvement_suggestions: list[str] = Field(default_factory=list)
    should_refine: bool

    @field_validator("missing_critical_terms", "improvement_suggestions", mode="before")
    @classmethod
    def parse_list_fields(cls, v: Any) -> list[str]:
        return _parse_stringified_list(v)


class SkillsGenerationResult(BaseModel):
    """Complete result for skills section generation."""

    reasoning: SkillsReasoning
    generated_skills: str = Field(description="Initial generated skills section")
    critique: SkillsCritique
    refinement_count: int = Field(default=0)
    final_skills: str = Field(description="Final skills section after refinements")


# =============================================================================
# CROSS-SECTION CONTEXT
# =============================================================================


class GenerationContext(BaseModel):
    """Context passed between sections for narrative coherence.

    Prevents keyword over-repetition and maintains consistent positioning.
    """

    # From summary generation
    established_identity: str = Field(
        default="", description="How candidate was positioned in summary"
    )
    key_metric_used: str = Field(default="", description="The headline metric from summary")
    primary_keywords_used: list[str] = Field(
        default_factory=list, description="Keywords already incorporated in summary"
    )

    # From experience generation (accumulated)
    total_bullets_generated: int = Field(default=0)
    metrics_used: list[str] = Field(
        default_factory=list, description="All metrics used across experiences"
    )
    skills_demonstrated: list[str] = Field(
        default_factory=list, description="Skills shown through experience bullets"
    )

    # Running totals for balance
    keyword_frequency: dict[str, int] = Field(
        default_factory=dict, description="How many times each keyword appears"
    )

    @field_validator("primary_keywords_used", "metrics_used", "skills_demonstrated", mode="before")
    @classmethod
    def parse_list_fields(cls, v: Any) -> list[str]:
        return _parse_stringified_list(v)


# =============================================================================
# COVER LETTER REASONING MODELS
# =============================================================================


class CoverLetterReasoning(BaseModel):
    """Intermediate reasoning for cover letter generation.

    Captures strategic thinking before writing the cover letter.
    """

    # Analysis phase
    opening_hook: str = Field(
        description="How to open compellingly - specific connection to company/role"
    )
    key_selling_points: list[str] = Field(
        description="Top 3 achievements/skills that best match job requirements"
    )
    strongest_alignment: str = Field(
        description="The single best match between candidate background and role"
    )
    company_connection: str = Field(
        description="Specific reason for interest in THIS company (not generic)"
    )

    # 2026 Tech Leadership qualities to highlight
    leadership_qualities: list[str] = Field(
        description=(
            "Which 2026 tech leader qualities to emphasize (pick 2-3): "
            "visionary leadership, adaptability, AI fluency, data-driven, "
            "collaboration, people development"
        )
    )
    problem_solution_framing: str = Field(
        description="How candidate's experience solves a specific company challenge/need"
    )

    # Strategy phase
    paragraph_structure: list[str] = Field(
        description="Planned content for each paragraph (3-4 paragraphs)"
    )
    keywords_to_incorporate: list[str] = Field(
        description="Job posting terms to weave in naturally (max 5)"
    )
    metric_to_feature: str = Field(description="Most impressive relevant metric from CV")
    call_to_action: str = Field(description="How to close with confidence (interview request)")

    # Constraints
    tone_guidance: str = Field(
        description="Tone to strike: confident but not arrogant, specific but not verbose"
    )
    aspects_to_avoid: list[str] = Field(
        default_factory=list,
        description="Topics/phrasings to avoid (salary, generic praise, desperation)",
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence in reasoning quality (0-1)"
    )

    @field_validator(
        "key_selling_points",
        "leadership_qualities",
        "paragraph_structure",
        "keywords_to_incorporate",
        "aspects_to_avoid",
        mode="before",
    )
    @classmethod
    def parse_list_fields(cls, v: Any) -> list[str]:
        return _parse_stringified_list(v)


class CoverLetterCritique(BaseModel):
    """Self-critique for generated cover letter."""

    # Quality checks
    has_compelling_opening: bool = Field(
        description="Does opening immediately establish relevance and interest?"
    )
    demonstrates_company_research: bool = Field(
        description="Shows specific knowledge of company, not generic statements?"
    )
    includes_quantified_achievement: bool = Field(
        description="Contains at least one hard metric from CV?"
    )
    demonstrates_problem_solution: bool = Field(
        default=True,
        description="Frames candidate's experience as solving a company challenge?",
    )
    shows_leadership_qualities: bool = Field(
        default=True,
        description="Conveys executive-level qualities (strategic vision, team building)?",
    )
    mirrors_job_keywords: bool = Field(
        description="Natural incorporation of 2-3 job posting terms?"
    )
    appropriate_length: bool = Field(description="Within specified character limit?")
    professional_tone: bool = Field(
        description="Confident but not arrogant, specific but not verbose?"
    )
    has_clear_call_to_action: bool = Field(description="Closes with clear next step request?")
    is_plain_text: bool = Field(description="No markdown, HTML, or special formatting?")

    # Overall assessment
    quality_level: QualityLevel = Field(description="Overall quality assessment")
    issues_found: list[str] = Field(
        default_factory=list, description="Specific issues that need fixing"
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list, description="Actionable suggestions for refinement"
    )
    should_refine: bool = Field(description="Does this need another iteration?")

    @field_validator("issues_found", "improvement_suggestions", mode="before")
    @classmethod
    def parse_list_fields(cls, v: Any) -> list[str]:
        return _parse_stringified_list(v)


class CoverLetterGenerationResult(BaseModel):
    """Complete result of cover letter generation with reasoning chain."""

    reasoning: CoverLetterReasoning
    generated_cover_letter: str = Field(description="Initial generated cover letter")
    critique: CoverLetterCritique | None = Field(
        default=None, description="Optional critique (skipped for speed in balanced mode)"
    )
    refinement_count: int = Field(default=0, description="Number of refinement iterations")
    final_cover_letter: str = Field(description="Final cover letter after any refinements")
    character_count: int = Field(description="Character count of final output")
