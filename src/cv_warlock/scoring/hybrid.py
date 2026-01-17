"""Hybrid scorer combining algorithmic and LLM-based assessment."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from cv_warlock.scoring.algorithmic import AlgorithmicScorer
from cv_warlock.scoring.models import (
    AlgorithmicScores,
    HybridMatchResult,
    LLMAssessmentOutput,
)

if TYPE_CHECKING:
    from cv_warlock.llm.base import LLMProvider
    from cv_warlock.models.cv import CVData
    from cv_warlock.models.job_spec import JobRequirements

logger = logging.getLogger(__name__)


# Prompt for LLM qualitative assessment
LLM_ASSESSMENT_PROMPT = """You are an expert technical recruiter providing qualitative assessment.

The algorithmic scoring has already computed these deterministic scores:

**Algorithmic Sub-Scores:**
- Exact skill match: {exact_match:.0%}
- Experience years fit: {years_fit:.0%}
- Education match: {edu_match:.0%}
- Recency score: {recency:.0%}

**Combined Algorithmic Score: {algo_score:.0%}**

=== CV DATA ===
{cv_data}

=== JOB REQUIREMENTS ===
{job_requirements}

=== YOUR TASK ===

Provide qualitative assessment that algorithms cannot capture:

1. **Transferable Skills**: Identify non-obvious skills that transfer to this role.
   Examples: leadership from non-tech roles, domain knowledge from adjacent industries,
   soft skills that map to job requirements.

2. **Contextual Strengths**: What narrative or story does this CV tell that matches the role?
   Consider career progression, company types, project scale, etc.

3. **Concerns**: Any red flags or concerns the algorithm might miss?
   Examples: job hopping pattern, skill depth vs breadth, overqualification.

4. **Adjustment Recommendation**: Based on your qualitative assessment, recommend an
   adjustment to the algorithmic score:
   - Range: -0.10 to +0.10
   - Positive if qualitative factors strengthen the match beyond what numbers show
   - Negative if there are concerns the algorithm missed
   - Zero if algorithmic score seems accurate

Be concise and actionable. Focus on insights the algorithm cannot provide."""


class _LLMAssessmentOutput(BaseModel):
    """Internal Pydantic model for structured LLM output."""

    transferable_skills: list[str] = Field(
        default_factory=list,
        description="Non-obvious transferable skills",
    )
    contextual_strengths: list[str] = Field(
        default_factory=list,
        description="Narrative strengths for this role",
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Red flags or concerns",
    )
    adjustment: float = Field(
        default=0.0,
        ge=-0.1,
        le=0.1,
        description="Score adjustment (-0.1 to +0.1)",
    )
    adjustment_rationale: str = Field(
        default="",
        description="Brief explanation for adjustment",
    )


class HybridScorer:
    """Combines algorithmic scoring with LLM qualitative assessment.

    Flow:
    1. Compute deterministic algorithmic sub-scores (fast, free, no API calls)
    2. Check knockout rules (auto-fail if required skills missing)
    3. Get LLM qualitative assessment
    4. Combine into final hybrid score
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
    ) -> None:
        """Initialize the hybrid scorer.

        Args:
            llm_provider: LLM provider for qualitative assessment.
        """
        self.llm_provider = llm_provider
        self.algorithmic = AlgorithmicScorer()

        # Set up LLM chain for qualitative assessment
        self._prompt = ChatPromptTemplate.from_template(LLM_ASSESSMENT_PROMPT)

    def score(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
    ) -> HybridMatchResult:
        """Compute hybrid match score.

        Args:
            cv_data: Parsed CV data.
            job_requirements: Parsed job requirements.

        Returns:
            HybridMatchResult with all scoring details.
        """
        # Step 1: Compute algorithmic scores (fast, no API calls)
        logger.info("Computing algorithmic scores...")
        algo_scores = self.algorithmic.compute(cv_data, job_requirements)

        # Step 2: Check knockout rule
        if algo_scores.knockout_triggered:
            logger.info(f"Knockout triggered: {algo_scores.knockout_reason}")
            return self._create_knockout_result(algo_scores)

        # Step 3: Get LLM qualitative assessment
        logger.info("Getting LLM qualitative assessment...")
        llm_assessment = self._get_llm_assessment(cv_data, job_requirements, algo_scores)

        # Step 4: Combine scores
        return self._combine_scores(cv_data, job_requirements, algo_scores, llm_assessment)

    def _create_knockout_result(
        self,
        algo_scores: AlgorithmicScores,
    ) -> HybridMatchResult:
        """Create result for knockout case (score = 0)."""
        return HybridMatchResult(
            strong_matches=[],
            partial_matches=[],
            gaps=[algo_scores.knockout_reason or "Missing required skills"],
            transferable_skills=[],
            relevance_score=0.0,
            score_breakdown=algo_scores.to_breakdown(),
            algorithmic_score=0.0,
            llm_adjustment=0.0,
            knockout_triggered=True,
            knockout_reason=algo_scores.knockout_reason,
            scoring_method="hybrid",
        )

    def _get_llm_assessment(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        algo_scores: AlgorithmicScores,
    ) -> LLMAssessmentOutput:
        """Get qualitative assessment from LLM."""
        try:
            # Prepare prompt variables
            prompt_vars = {
                "exact_match": algo_scores.exact_skill_match,
                "years_fit": algo_scores.experience_years_fit,
                "edu_match": algo_scores.education_match,
                "recency": algo_scores.recency_score,
                "algo_score": algo_scores.total,
                "cv_data": self._serialize_cv(cv_data),
                "job_requirements": self._serialize_job(job_requirements),
            }

            # Get structured output from LLM
            model = self.llm_provider.get_extraction_model()
            chain = self._prompt | model.with_structured_output(
                _LLMAssessmentOutput,
                method="function_calling",
            )

            result = chain.invoke(prompt_vars)

            return LLMAssessmentOutput(
                transferable_skills=result.transferable_skills,
                contextual_strengths=result.contextual_strengths,
                concerns=result.concerns,
                adjustment=result.adjustment,
                adjustment_rationale=result.adjustment_rationale,
            )

        except Exception as e:
            logger.warning(f"LLM assessment failed, using neutral adjustment: {e}")
            return LLMAssessmentOutput(
                transferable_skills=[],
                contextual_strengths=[],
                concerns=[f"LLM assessment failed: {e!s}"],
                adjustment=0.0,
                adjustment_rationale="Failed to get LLM assessment",
            )

    def _combine_scores(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        algo_scores: AlgorithmicScores,
        llm_assessment: LLMAssessmentOutput,
    ) -> HybridMatchResult:
        """Combine algorithmic and LLM scores into final result."""
        # Apply LLM adjustment to algorithmic score
        final_score = max(0.0, min(1.0, algo_scores.total + llm_assessment.adjustment))

        # Build strong/partial matches lists from algorithmic analysis
        strong_matches, partial_matches = self._categorize_matches(
            cv_data, job_requirements
        )

        # Identify gaps (missing required skills)
        gaps = self._identify_gaps(cv_data, job_requirements)

        return HybridMatchResult(
            strong_matches=strong_matches,
            partial_matches=partial_matches,
            gaps=gaps,
            transferable_skills=llm_assessment.transferable_skills,
            relevance_score=final_score,
            score_breakdown=algo_scores.to_breakdown(),
            algorithmic_score=algo_scores.total,
            llm_adjustment=llm_assessment.adjustment,
            knockout_triggered=False,
            knockout_reason=None,
            scoring_method="hybrid",
        )

    def _categorize_matches(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
    ) -> tuple[list[str], list[str]]:
        """Categorize skill matches into strong and partial."""
        cv_skills = {s.lower() for s in cv_data.skills}
        for exp in cv_data.experiences:
            cv_skills.update(s.lower() for s in exp.skills_used)

        # Also check experience text
        exp_text = ""
        for exp in cv_data.experiences:
            exp_text += f" {exp.title} {exp.description} {' '.join(exp.achievements)}"
        exp_text = exp_text.lower()

        required = job_requirements.required_skills
        preferred = job_requirements.preferred_skills

        strong_matches: list[str] = []
        partial_matches: list[str] = []

        # Check required skills
        for skill in required:
            skill_lower = skill.lower()
            if skill_lower in cv_skills:
                strong_matches.append(f"{skill} (required, exact match)")
            elif skill_lower in exp_text:
                partial_matches.append(f"{skill} (required, mentioned in experience)")

        # Check preferred skills
        for skill in preferred:
            skill_lower = skill.lower()
            if skill_lower in cv_skills:
                strong_matches.append(f"{skill} (preferred, exact match)")
            elif skill_lower in exp_text:
                partial_matches.append(f"{skill} (preferred, mentioned in experience)")

        return strong_matches, partial_matches

    def _identify_gaps(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
    ) -> list[str]:
        """Identify missing required skills."""
        cv_skills = {s.lower() for s in cv_data.skills}
        for exp in cv_data.experiences:
            cv_skills.update(s.lower() for s in exp.skills_used)

        # Also check experience text
        exp_text = ""
        for exp in cv_data.experiences:
            exp_text += f" {exp.title} {exp.description} {' '.join(exp.achievements)}"
        exp_text = exp_text.lower()

        gaps: list[str] = []
        for skill in job_requirements.required_skills:
            skill_lower = skill.lower()
            if skill_lower not in cv_skills and skill_lower not in exp_text:
                gaps.append(skill)

        return gaps

    @staticmethod
    def _serialize_cv(cv_data: CVData) -> str:
        """Serialize CV data for prompt."""
        return json.dumps(cv_data, indent=2, default=str)

    @staticmethod
    def _serialize_job(job_requirements: JobRequirements) -> str:
        """Serialize job requirements for prompt."""
        return json.dumps(job_requirements, indent=2, default=str)
