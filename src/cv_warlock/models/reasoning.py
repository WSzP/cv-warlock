"""Chain-of-thought reasoning models for CV generation.

These models capture intermediate reasoning steps, quality critiques,
and generation results for each CV section. This enables:
- Transparent reasoning that can be audited/debugged
- Self-critique and refinement loops
- Cross-section context for coherent output
"""

from enum import Enum

from pydantic import BaseModel, Field


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


class ExperienceGenerationResult(BaseModel):
    """Complete result for one experience entry's tailoring."""

    experience_title: str
    experience_company: str
    reasoning: ExperienceReasoning
    generated_bullets: list[str] = Field(description="Initial generated bullets")
    critique: ExperienceCritique
    refinement_count: int = Field(default=0)
    final_bullets: list[str] = Field(description="Final bullets after refinements")


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
    ordering_rationale: str = Field(
        default="", description="Why skills are ordered this way"
    )

    # Omissions
    skills_to_omit: list[str] = Field(
        default_factory=list, description="Candidate skills irrelevant to this role"
    )


class SkillsCritique(BaseModel):
    """Self-critique for skills section ATS optimization."""

    # ATS optimization checks
    all_required_skills_present: bool = Field(
        description="All required skills (that candidate has) included?"
    )
    uses_exact_job_terminology: bool = Field(
        description="Uses exact wording from job posting?"
    )
    appropriate_categorization: bool = Field(
        description="Skills grouped logically matching job structure?"
    )
    no_irrelevant_skills: bool = Field(
        description="No skills that dilute relevance signal?"
    )
    no_fabricated_skills: bool = Field(
        description="No skills candidate doesn't actually have?"
    )

    # Quality assessment
    quality_level: QualityLevel
    missing_critical_terms: list[str] = Field(
        default_factory=list, description="Important terms that should be added"
    )
    improvement_suggestions: list[str] = Field(default_factory=list)
    should_refine: bool


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
