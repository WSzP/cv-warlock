"""CV tailoring processor with chain-of-thought reasoning.

This module provides CV section tailoring with optional CoT reasoning.
When CoT is enabled (default), each section follows a REASON -> GENERATE pattern
for balanced quality and speed.

Performance optimizations:
- Balanced mode: 2 LLM calls per section (REASON→GENERATE)
- Compressed context: Only essential reasoning fields passed to generation
- Async experience processing: All experiences processed in parallel
"""

from concurrent.futures import ThreadPoolExecutor

from langchain_core.prompts import ChatPromptTemplate

from cv_warlock.config import get_settings
from cv_warlock.llm.base import LLMProvider
from cv_warlock.models.cv import CVData, Experience
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.models.reasoning import (
    BatchExperienceReasoning,
    ExperienceCritique,
    ExperienceGenerationResult,
    ExperienceReasoning,
    GenerationContext,
    QualityLevel,
    SkillsCritique,
    SkillsGenerationResult,
    SkillsReasoning,
    SummaryCritique,
    SummaryGenerationResult,
    SummaryReasoning,
)
from cv_warlock.models.state import TailoringPlan
from cv_warlock.prompts.generation import (
    CV_ASSEMBLY_PROMPT,
    EXPERIENCE_TAILORING_PROMPT,
    SKILLS_TAILORING_PROMPT,
    SUMMARY_TAILORING_PROMPT,
)
from cv_warlock.prompts.reasoning import (
    BATCH_EXPERIENCE_REASONING_PROMPT,
    EXPERIENCE_GENERATION_PROMPT,
    EXPERIENCE_REASONING_PROMPT,
    SKILLS_GENERATION_PROMPT,
    SKILLS_REASONING_PROMPT,
    SUMMARY_GENERATION_PROMPT,
    SUMMARY_REASONING_PROMPT,
)
from cv_warlock.utils.date_utils import should_tailor_experience


def _get_relevant_skills_for_experience(
    experience: Experience,
    job_requirements: JobRequirements,
    max_skills: int = 7,
) -> list[str]:
    """Extract only job skills relevant to this specific experience.

    Uses fuzzy matching to find skills mentioned in experience text,
    reducing token usage by ~1000 tokens per experience call.

    Args:
        experience: The experience entry to analyze.
        job_requirements: Full job requirements.
        max_skills: Maximum skills to return.

    Returns:
        List of relevant skills from the job requirements.
    """
    # Build searchable text from experience
    exp_text = " ".join(
        [
            experience.title.lower(),
            experience.company.lower(),
            experience.description.lower() if experience.description else "",
            " ".join(a.lower() for a in experience.achievements),
            " ".join(s.lower() for s in experience.skills_used),
        ]
    )

    relevant_skills: list[str] = []

    # Check required skills first (higher priority)
    for skill in job_requirements.required_skills:
        skill_lower = skill.lower()
        # Check for exact match or partial match (e.g., "Python" in "Python development")
        if skill_lower in exp_text or any(word in exp_text for word in skill_lower.split()):
            relevant_skills.append(skill)

    # Then check preferred skills
    for skill in job_requirements.preferred_skills:
        if len(relevant_skills) >= max_skills:
            break
        skill_lower = skill.lower()
        if skill_lower in exp_text or any(word in exp_text for word in skill_lower.split()):
            if skill not in relevant_skills:
                relevant_skills.append(skill)

    # If we found very few matches, include top required skills anyway
    # (the experience should still reference key job requirements)
    if len(relevant_skills) < 3:
        for skill in job_requirements.required_skills[:5]:
            if skill not in relevant_skills:
                relevant_skills.append(skill)
            if len(relevant_skills) >= max_skills:
                break

    return relevant_skills[:max_skills]


def _compress_summary_reasoning(reasoning: SummaryReasoning) -> str:
    """Extract essential fields from summary reasoning for generation prompt.

    Instead of passing full JSON (~800 tokens), pass only what's needed (~150 tokens).
    """
    return f"""Hook: {reasoning.hook_strategy}
Keywords: {", ".join(reasoning.key_keywords_to_include)}
Metric: {reasoning.strongest_metric}
Differentiator: {reasoning.unique_differentiator}
Value proposition: {reasoning.value_proposition}
Fit statement: {reasoning.fit_statement}"""


def _compress_experience_reasoning(reasoning: ExperienceReasoning) -> str:
    """Extract essential fields from experience reasoning for generation prompt."""
    bullets_summary = ""
    if reasoning.bullet_reasoning:
        bullets_summary = "\n".join(
            f"- {br.power_verb_choice}: {br.reframed_bullet[:80]}..."
            for br in reasoning.bullet_reasoning[:5]
        )

    return f"""Relevance: {reasoning.relevance_score:.1f}
Strategy: {reasoning.emphasis_strategy}
Keywords: {", ".join(reasoning.keywords_to_incorporate[:5])}
Priority achievements: {", ".join(reasoning.achievements_to_prioritize[:3])}
Bullet plans:
{bullets_summary}"""


def _compress_skills_reasoning(reasoning: SkillsReasoning) -> str:
    """Extract essential fields from skills reasoning for generation prompt."""
    categories = "\n".join(
        f"- {cat}: {', '.join(skills[:5])}"
        for cat, skills in list(reasoning.category_groupings.items())[:5]
    )

    return f"""Required matched: {", ".join(reasoning.required_skills_matched[:7])}
Preferred matched: {", ".join(reasoning.preferred_skills_matched[:5])}
Dual format: {", ".join(reasoning.dual_format_terms[:5])}
Categories:
{categories}"""


def _create_placeholder_summary_critique() -> SummaryCritique:
    """Create a placeholder critique for balanced mode (no actual critique step)."""
    return SummaryCritique(
        has_strong_opening_hook=True,
        includes_quantified_achievement=True,
        mirrors_job_keywords=True,
        appropriate_length=True,
        avoids_fluff=True,
        quality_level=QualityLevel.GOOD,
        issues_found=[],
        improvement_suggestions=[],
        should_refine=False,
    )


def _create_placeholder_experience_critique() -> ExperienceCritique:
    """Create a placeholder critique for balanced mode (no actual critique step)."""
    return ExperienceCritique(
        all_bullets_start_with_power_verb=True,
        all_bullets_show_impact=True,
        metrics_present_where_possible=True,
        relevant_keywords_incorporated=True,
        bullets_appropriately_ordered=True,
        quality_level=QualityLevel.GOOD,
        weak_bullets=[],
        improvement_suggestions=[],
        should_refine=False,
    )


def _create_placeholder_skills_critique() -> SkillsCritique:
    """Create a placeholder critique for balanced mode (no actual critique step)."""
    return SkillsCritique(
        all_required_skills_present=True,
        uses_exact_job_terminology=True,
        appropriate_categorization=True,
        no_irrelevant_skills=True,
        no_fabricated_skills=True,
        quality_level=QualityLevel.GOOD,
        missing_critical_terms=[],
        improvement_suggestions=[],
        should_refine=False,
    )


class CVTailor:
    """Tailor CV sections with optional chain-of-thought reasoning.

    When use_cot=True (default), uses a REASON -> GENERATE pipeline.
    When use_cot=False, uses direct single-prompt generation (faster but lower quality).
    """

    def __init__(self, llm_provider: LLMProvider, use_cot: bool = True):
        """Initialize tailor with optional CoT reasoning.

        Args:
            llm_provider: LLM provider instance.
            use_cot: Whether to use chain-of-thought reasoning (default True).
                     Set to False for faster generation with lower quality.
        """
        self.llm_provider = llm_provider
        self.use_cot = use_cot

        # Direct mode prompts
        self.summary_prompt = ChatPromptTemplate.from_template(SUMMARY_TAILORING_PROMPT)
        self.experience_prompt = ChatPromptTemplate.from_template(EXPERIENCE_TAILORING_PROMPT)
        self.skills_prompt = ChatPromptTemplate.from_template(SKILLS_TAILORING_PROMPT)
        self.assembly_prompt = ChatPromptTemplate.from_template(CV_ASSEMBLY_PROMPT)

        # CoT prompts - Summary
        self.summary_reasoning_prompt = ChatPromptTemplate.from_template(SUMMARY_REASONING_PROMPT)
        self.summary_gen_prompt = ChatPromptTemplate.from_template(SUMMARY_GENERATION_PROMPT)

        # CoT prompts - Experience
        self.exp_reasoning_prompt = ChatPromptTemplate.from_template(EXPERIENCE_REASONING_PROMPT)
        self.batch_exp_reasoning_prompt = ChatPromptTemplate.from_template(
            BATCH_EXPERIENCE_REASONING_PROMPT
        )
        self.exp_gen_prompt = ChatPromptTemplate.from_template(EXPERIENCE_GENERATION_PROMPT)

        # CoT prompts - Skills
        self.skills_reasoning_prompt = ChatPromptTemplate.from_template(SKILLS_REASONING_PROMPT)
        self.skills_gen_prompt = ChatPromptTemplate.from_template(SKILLS_GENERATION_PROMPT)

    # =========================================================================
    # SUMMARY TAILORING
    # =========================================================================

    def tailor_summary(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
        tailored_skills_preview: str = "",
    ) -> str:
        """Tailor the professional summary (LAST in pipeline).

        Uses CoT reasoning if enabled, otherwise direct generation.

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.
            tailoring_plan: Tailoring plan.
            tailored_skills_preview: The tailored skills section for reference.

        Returns:
            str: Tailored summary text.
        """
        if self.use_cot:
            result = self.tailor_summary_with_cot(
                cv_data,
                job_requirements,
                tailoring_plan,
                tailored_skills_preview=tailored_skills_preview,
            )
            return result.final_summary
        else:
            return self._tailor_summary_direct(
                cv_data,
                job_requirements,
                tailoring_plan,
                tailored_skills_preview=tailored_skills_preview,
            )

    def _tailor_summary_direct(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
        tailored_skills_preview: str = "",
    ) -> str:
        """Direct summary generation without CoT (summary is LAST in pipeline)."""
        model = self.llm_provider.get_chat_model()
        chain = self.summary_prompt | model
        result = chain.invoke(
            {
                "original_summary": cv_data.summary or "No summary provided",
                "job_title": job_requirements.job_title,
                "company": job_requirements.company or "the company",
                "key_requirements": ", ".join(job_requirements.required_skills[:5]),
                "relevant_strengths": ", ".join(tailoring_plan["skills_to_highlight"][:5]),
                "tailored_skills_preview": tailored_skills_preview or "Not yet generated",
            }
        )
        return result.content

    def tailor_summary_with_cot(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
        context: GenerationContext | None = None,
        tailored_skills_preview: str = "",
    ) -> SummaryGenerationResult:
        """Tailor summary with chain-of-thought reasoning (LAST in pipeline).

        Balanced mode: REASON → GENERATE only (2 LLM calls).
        Summary is last, so it can reference tailored skills and experiences.

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.
            tailoring_plan: Tailoring plan.
            context: Optional context from previous sections.
            tailored_skills_preview: The tailored skills section for reference.

        Returns:
            SummaryGenerationResult with reasoning and generation output.
        """
        # Step 1: REASON
        reasoning = self._reason_summary(cv_data, job_requirements, tailoring_plan)

        # Step 2: GENERATE (summary is LAST, can reference tailored skills)
        generated = self._generate_summary_from_reasoning(
            reasoning, job_requirements, tailored_skills_preview
        )

        return SummaryGenerationResult(
            reasoning=reasoning,
            generated_summary=generated,
            critique=_create_placeholder_summary_critique(),
            refinement_count=0,
            final_summary=generated,
        )

    def _reason_summary(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
    ) -> SummaryReasoning:
        """Generate reasoning for summary (Step 1)."""
        model = self.llm_provider.get_extraction_model()
        structured_model = model.with_structured_output(SummaryReasoning, method="function_calling")

        chain = self.summary_reasoning_prompt | structured_model
        return chain.invoke(
            {
                "original_summary": cv_data.summary or "No summary provided",
                "job_title": job_requirements.job_title,
                "company": job_requirements.company or "the company",
                "key_requirements": ", ".join(job_requirements.required_skills[:7]),
                "relevant_strengths": ", ".join(tailoring_plan["skills_to_highlight"][:7]),
                "tailoring_plan_summary": f"""
Focus areas: {", ".join(tailoring_plan["summary_focus"][:3])}
Key achievements to feature: {", ".join(tailoring_plan["achievements_to_feature"][:3])}
Keywords to incorporate: {", ".join(tailoring_plan["keywords_to_incorporate"][:5])}
""",
            }
        )

    def _generate_summary_from_reasoning(
        self,
        reasoning: SummaryReasoning,
        job_requirements: JobRequirements,
        tailored_skills_preview: str = "",
    ) -> str:
        """Generate summary based on reasoning (Step 2).

        Uses compressed reasoning context (~150 tokens) instead of full JSON (~800 tokens).
        Summary is LAST in pipeline, so can reference tailored skills.
        """
        model = self.llm_provider.get_chat_model()

        chain = self.summary_gen_prompt | model
        result = chain.invoke(
            {
                # Use compressed context instead of full JSON dump
                "reasoning_json": _compress_summary_reasoning(reasoning),
                "strongest_metric": reasoning.strongest_metric,
                "keywords": ", ".join(reasoning.key_keywords_to_include),
                "tailored_skills_preview": tailored_skills_preview or "Not yet generated",
            }
        )
        return result.content

    # =========================================================================
    # EXPERIENCE TAILORING
    # =========================================================================

    def tailor_experience(
        self,
        experience: Experience,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
    ) -> str:
        """Tailor a single experience entry.

        Args:
            experience: Experience entry to tailor.
            job_requirements: Structured job requirements.
            tailoring_plan: Tailoring plan.

        Returns:
            str: Tailored experience with header and bullets as text.
        """
        header = self._format_experience_header(experience)
        if self.use_cot:
            result = self.tailor_experience_with_cot(experience, job_requirements, tailoring_plan)
            bullets = "\n".join(f"- {b}" for b in result.final_bullets)
        else:
            bullets = self._tailor_experience_direct(experience, job_requirements, tailoring_plan)
        return f"{header}\n{bullets}"

    def _tailor_experience_direct(
        self,
        experience: Experience,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
    ) -> str:
        """Direct experience generation without CoT."""
        model = self.llm_provider.get_chat_model()
        chain = self.experience_prompt | model
        result = chain.invoke(
            {
                "title": experience.title,
                "company": experience.company,
                "period": f"{experience.start_date} - {experience.end_date or 'Present'}",
                "description": experience.description,
                "achievements": "\n".join(f"- {a}" for a in experience.achievements),
                "target_requirements": ", ".join(job_requirements.required_skills[:5]),
                "skills_to_emphasize": ", ".join(tailoring_plan["skills_to_highlight"][:5]),
            }
        )
        return result.content

    def tailor_experience_with_cot(
        self,
        experience: Experience,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
        context: GenerationContext | None = None,
    ) -> ExperienceGenerationResult:
        """Tailor experience with chain-of-thought reasoning (balanced mode).

        Balanced mode: REASON → GENERATE only (2 LLM calls per experience).
        Skips CRITIQUE → REFINE for faster execution while maintaining quality.

        Args:
            experience: Experience entry to tailor.
            job_requirements: Structured job requirements.
            tailoring_plan: Tailoring plan.
            context: Optional context from previous sections.

        Returns:
            ExperienceGenerationResult with reasoning and generation output.
        """
        # Step 1: REASON
        reasoning = self._reason_experience(experience, job_requirements, tailoring_plan, context)

        # Determine bullet count from emphasis strategy
        bullet_count = self._get_bullet_count(reasoning.emphasis_strategy)

        # Step 2: GENERATE
        generated_text = self._generate_experience_from_reasoning(
            reasoning, job_requirements, bullet_count
        )
        generated_bullets = self._parse_bullets(generated_text)

        return ExperienceGenerationResult(
            experience_title=experience.title,
            experience_company=experience.company,
            reasoning=reasoning,
            generated_bullets=generated_bullets,
            critique=_create_placeholder_experience_critique(),
            refinement_count=0,
            final_bullets=generated_bullets,
        )

    def _reason_experience(
        self,
        experience: Experience,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
        context: GenerationContext | None,
    ) -> ExperienceReasoning:
        """Generate reasoning for experience (Step 1).

        Uses targeted skill matching to reduce token usage - only sends
        skills relevant to this specific experience instead of all requirements.
        """
        model = self.llm_provider.get_extraction_model()
        structured_model = model.with_structured_output(
            ExperienceReasoning, method="function_calling"
        )

        # Get only skills relevant to THIS experience (saves ~1000 tokens per call)
        relevant_skills = _get_relevant_skills_for_experience(
            experience, job_requirements, max_skills=7
        )

        chain = self.exp_reasoning_prompt | structured_model
        return chain.invoke(
            {
                "title": experience.title,
                "company": experience.company,
                "period": f"{experience.start_date} - {experience.end_date or 'Present'}",
                "description": experience.description or "No description provided",
                "achievements": "\n".join(f"- {a}" for a in experience.achievements)
                if experience.achievements
                else "No specific achievements listed",
                "job_title": job_requirements.job_title,
                "target_requirements": ", ".join(relevant_skills),
                "skills_to_emphasize": ", ".join(tailoring_plan["skills_to_highlight"][:5]),
                "established_identity": context.established_identity
                if context
                else "Not yet established",
                "keywords_already_used": ", ".join(context.primary_keywords_used)
                if context
                else "None yet",
                "metrics_already_used": ", ".join(context.metrics_used) if context else "None yet",
            }
        )

    def _batch_reason_experiences(
        self,
        experiences: list[tuple[int, Experience]],
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
        context: GenerationContext | None,
    ) -> dict[int, ExperienceReasoning]:
        """Generate reasoning for ALL experiences in a single LLM call.

        This is a major optimization: instead of N separate REASON calls (one per experience),
        we make 1 call that returns reasoning for all experiences. This reduces API latency
        significantly since each call has ~500ms overhead.

        Args:
            experiences: List of (index, Experience) tuples to reason about.
            job_requirements: Job requirements for the target role.
            tailoring_plan: Overall tailoring strategy.
            context: Optional context from previous sections.

        Returns:
            Dict mapping experience index to its reasoning.
        """
        if not experiences:
            return {}

        model = self.llm_provider.get_extraction_model()
        structured_model = model.with_structured_output(
            BatchExperienceReasoning, method="function_calling"
        )

        # Format all experiences for the batch prompt
        experiences_text_parts = []
        for idx, exp in experiences:
            achievements_text = (
                "\n".join(f"  - {a}" for a in exp.achievements)
                if exp.achievements
                else "  No specific achievements listed"
            )
            exp_text = f"""
--- Experience {idx} ---
Title: {exp.title}
Company: {exp.company}
Period: {exp.start_date} - {exp.end_date or "Present"}
Description: {exp.description or "No description provided"}
Achievements:
{achievements_text}
"""
            experiences_text_parts.append(exp_text)

        experiences_text = "\n".join(experiences_text_parts)

        # Get all relevant skills for the batch
        all_required = job_requirements.required_skills[:10]
        all_preferred = job_requirements.preferred_skills[:5]
        target_requirements = ", ".join(all_required + all_preferred)

        chain = self.batch_exp_reasoning_prompt | structured_model
        batch_result: BatchExperienceReasoning = chain.invoke(
            {
                "job_title": job_requirements.job_title,
                "target_requirements": target_requirements,
                "skills_to_emphasize": ", ".join(tailoring_plan["skills_to_highlight"][:5]),
                "experiences_text": experiences_text,
                "established_identity": context.established_identity
                if context
                else "Not yet established",
                "keywords_already_used": ", ".join(context.primary_keywords_used)
                if context
                else "None yet",
                "metrics_already_used": ", ".join(context.metrics_used) if context else "None yet",
            }
        )

        # Map results back to experience indices
        result_map: dict[int, ExperienceReasoning] = {}
        for single_reasoning in batch_result.experiences:
            result_map[single_reasoning.experience_index] = single_reasoning.reasoning

        return result_map

    def _get_bullet_count(self, emphasis_strategy: str) -> int:
        """Determine bullet count from emphasis strategy.

        Best practice: 4-5 bullets for recent major roles, fewer for older ones.
        Keep CV scannable - less is more.
        """
        strategy_upper = emphasis_strategy.upper()
        if "HIGH" in strategy_upper:
            return 5  # Cornerstone experience
        elif "MED" in strategy_upper:
            return 3  # Focused coverage
        else:
            return 2  # Minimal coverage

    def _generate_experience_from_reasoning(
        self,
        reasoning: ExperienceReasoning,
        job_requirements: JobRequirements,
        bullet_count: int,
    ) -> str:
        """Generate experience bullets from reasoning (Step 2).

        Uses compressed reasoning context (~200 tokens) instead of full JSON (~1000 tokens).
        """
        model = self.llm_provider.get_chat_model()

        chain = self.exp_gen_prompt | model
        result = chain.invoke(
            {
                # Use compressed context instead of full JSON dump
                "reasoning_json": _compress_experience_reasoning(reasoning),
                "bullet_count": bullet_count,
                "keywords_to_use": ", ".join(reasoning.keywords_to_incorporate[:5]),
            }
        )
        return result.content

    def _parse_bullets(self, text: str) -> list[str]:
        """Parse bullet points from generated text."""
        bullets = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                bullets.append(line[2:].strip())
            elif line.startswith("* "):
                bullets.append(line[2:].strip())
            elif line and not line.startswith("#"):
                # Handle lines without bullet prefix
                bullets.append(line)
        return [b for b in bullets if b]

    def tailor_experiences(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
        lookback_years: int | None = None,
    ) -> list[str]:
        """Tailor experience entries with lookback filtering.

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.
            tailoring_plan: Tailoring plan.
            lookback_years: Only tailor jobs ending within this many years.
                           If None, uses settings default.

        Returns:
            list[str]: List of tailored experience texts.
        """
        if lookback_years is None:
            lookback_years = get_settings().lookback_years

        tailored = []
        for exp in cv_data.experiences:
            if should_tailor_experience(exp.end_date, lookback_years):
                # Recent job - tailor it
                tailored_exp = self.tailor_experience(exp, job_requirements, tailoring_plan)
            else:
                # Old job - pass through original bullets unchanged
                tailored_exp = self._format_passthrough_experience(exp)
            tailored.append(tailored_exp)
        return tailored

    def _format_experience_header(self, exp: Experience) -> str:
        """Format the experience header (title, company, dates).

        This ensures job metadata is preserved in the tailored output,
        preventing the assembly prompt from inventing headers.
        """
        period = f"{exp.start_date} - {exp.end_date or 'Present'}"
        return f"### {exp.title} | {exp.company}\n*{period}*"

    def _format_passthrough_experience(self, exp: Experience) -> str:
        """Format an experience as-is without tailoring (for old jobs)."""
        header = self._format_experience_header(exp)
        if exp.achievements:
            bullets = "\n".join(f"- {a}" for a in exp.achievements)
        elif exp.description:
            bullets = f"- {exp.description}"
        else:
            bullets = "- No description provided"
        return f"{header}\n{bullets}"

    def tailor_experiences_with_cot(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
        context: GenerationContext | None = None,
        lookback_years: int | None = None,
    ) -> tuple[list[str], list[ExperienceGenerationResult], GenerationContext]:
        """Tailor experiences with CoT and lookback filtering, using batch reasoning.

        OPTIMIZATION: Uses batch reasoning to process all experiences in 1 LLM call
        instead of N separate calls, then generates bullets in parallel.

        Previous: N×2 calls (N reason + N generate)
        Now: 1+N calls (1 batch reason + N parallel generate)

        For 5 experiences: 10 calls → 6 calls = 40% reduction in API calls.

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.
            tailoring_plan: Tailoring plan.
            context: Optional context from skills section.
            lookback_years: Only tailor jobs ending within this many years.
                           If None, uses settings default.

        Returns:
            Tuple of (tailored texts, generation results, updated context).
        """
        if lookback_years is None:
            lookback_years = get_settings().lookback_years

        current_context = context or GenerationContext()

        # Separate experiences into tailorable and passthrough
        experiences_to_tailor: list[tuple[int, Experience]] = []
        passthrough_indices: list[int] = []

        for i, exp in enumerate(cv_data.experiences):
            if should_tailor_experience(exp.end_date, lookback_years):
                experiences_to_tailor.append((i, exp))
            else:
                passthrough_indices.append(i)

        tailor_results: list[ExperienceGenerationResult] = []
        if experiences_to_tailor:
            # BATCH REASONING: Get all reasoning in 1 LLM call
            reasoning_map = self._batch_reason_experiences(
                experiences_to_tailor, job_requirements, tailoring_plan, current_context
            )

            # PARALLEL GENERATION: Generate bullets for each experience in parallel
            def generate_from_reasoning(
                idx_exp: tuple[int, Experience],
            ) -> ExperienceGenerationResult:
                idx, exp = idx_exp
                reasoning = reasoning_map.get(idx)

                # Fallback to individual reasoning if batch missed this experience
                if reasoning is None:
                    reasoning = self._reason_experience(
                        exp, job_requirements, tailoring_plan, current_context
                    )

                bullet_count = self._get_bullet_count(reasoning.emphasis_strategy)
                generated_text = self._generate_experience_from_reasoning(
                    reasoning, job_requirements, bullet_count
                )
                generated_bullets = self._parse_bullets(generated_text)

                return ExperienceGenerationResult(
                    experience_title=exp.title,
                    experience_company=exp.company,
                    reasoning=reasoning,
                    generated_bullets=generated_bullets,
                    critique=_create_placeholder_experience_critique(),
                    refinement_count=0,
                    final_bullets=generated_bullets,
                )

            # Use ThreadPoolExecutor for parallel generation calls
            with ThreadPoolExecutor(max_workers=None) as executor:
                tailor_results = list(executor.map(generate_from_reasoning, experiences_to_tailor))

        # Build results maintaining original order
        # Map tailor results back to their original indices
        tailor_result_map = {
            idx: result for (idx, _), result in zip(experiences_to_tailor, tailor_results)
        }

        # Build combined outputs in original order
        all_tailored_texts: list[str] = []
        all_results: list[ExperienceGenerationResult] = []

        for i, exp in enumerate(cv_data.experiences):
            if i in tailor_result_map:
                # Tailored experience
                result = tailor_result_map[i]
                header = self._format_experience_header(exp)
                bullets = "\n".join(f"- {b}" for b in result.final_bullets)
                tailored_text = f"{header}\n{bullets}"
                all_tailored_texts.append(tailored_text)
                all_results.append(result)

                # Update context
                current_context.total_bullets_generated += len(result.final_bullets)
                if result.reasoning.bullet_reasoning:
                    for br in result.reasoning.bullet_reasoning:
                        if br.metric_identified:
                            current_context.metrics_used.append(br.metric_identified)
                current_context.skills_demonstrated.extend(
                    result.reasoning.transferable_skills_identified
                )
                for kw in result.reasoning.keywords_to_incorporate:
                    current_context.keyword_frequency[kw] = (
                        current_context.keyword_frequency.get(kw, 0) + 1
                    )
            else:
                # Passthrough experience (outside lookback window)
                passthrough_text = self._format_passthrough_experience(exp)
                all_tailored_texts.append(passthrough_text)
                # Create a placeholder result for passthrough
                all_results.append(self._create_passthrough_result(exp))

        return all_tailored_texts, all_results, current_context

    def _create_passthrough_result(self, exp: Experience) -> ExperienceGenerationResult:
        """Create a placeholder result for experiences outside lookback window."""
        original_bullets = exp.achievements if exp.achievements else [exp.description or ""]

        return ExperienceGenerationResult(
            experience_title=exp.title,
            experience_company=exp.company,
            reasoning=ExperienceReasoning(
                relevance_score=0.0,
                emphasis_strategy="PASSTHROUGH - Outside lookback window",
                keywords_to_incorporate=[],
                achievements_to_prioritize=[],
                transferable_skills_identified=[],
                bullet_reasoning=[],
            ),
            generated_bullets=original_bullets,
            critique=_create_placeholder_experience_critique(),
            refinement_count=0,
            final_bullets=original_bullets,
        )

    # =========================================================================
    # SKILLS TAILORING
    # =========================================================================

    def tailor_skills(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
    ) -> str:
        """Tailor the skills section.

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.

        Returns:
            str: Tailored skills section text.
        """
        if self.use_cot:
            result = self.tailor_skills_with_cot(cv_data, job_requirements)
            return result.final_skills
        else:
            return self._tailor_skills_direct(cv_data, job_requirements)

    def _tailor_skills_direct(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
    ) -> str:
        """Direct skills generation without CoT."""
        model = self.llm_provider.get_chat_model()
        chain = self.skills_prompt | model
        result = chain.invoke(
            {
                "all_skills": ", ".join(cv_data.skills),
                "required_skills": ", ".join(job_requirements.required_skills),
                "preferred_skills": ", ".join(job_requirements.preferred_skills),
            }
        )
        return result.content

    def tailor_skills_with_cot(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        context: GenerationContext | None = None,
    ) -> SkillsGenerationResult:
        """Tailor skills with chain-of-thought reasoning (balanced mode).

        Balanced mode: REASON → GENERATE only (2 LLM calls).
        Skips CRITIQUE → REFINE for faster execution while maintaining quality.

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.
            context: Optional context from previous sections.

        Returns:
            SkillsGenerationResult with reasoning and generation output.
        """
        # Step 1: REASON
        reasoning = self._reason_skills(cv_data, job_requirements, context)

        # Step 2: GENERATE
        generated = self._generate_skills_from_reasoning(reasoning)

        return SkillsGenerationResult(
            reasoning=reasoning,
            generated_skills=generated,
            critique=_create_placeholder_skills_critique(),
            refinement_count=0,
            final_skills=generated,
        )

    def _reason_skills(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        context: GenerationContext | None,
    ) -> SkillsReasoning:
        """Generate reasoning for skills section (Step 1)."""
        model = self.llm_provider.get_extraction_model()
        structured_model = model.with_structured_output(SkillsReasoning, method="function_calling")

        chain = self.skills_reasoning_prompt | structured_model
        return chain.invoke(
            {
                "all_skills": ", ".join(cv_data.skills),
                "required_skills": ", ".join(job_requirements.required_skills),
                "preferred_skills": ", ".join(job_requirements.preferred_skills),
                "skills_from_experience": ", ".join(context.skills_demonstrated)
                if context
                else "Not yet analyzed",
                "keywords_used": ", ".join(
                    k for k, v in (context.keyword_frequency if context else {}).items() if v >= 2
                )
                or "None heavily used yet",
            }
        )

    def _generate_skills_from_reasoning(self, reasoning: SkillsReasoning) -> str:
        """Generate skills section from reasoning (Step 2).

        Uses compressed reasoning context (~150 tokens) instead of full JSON (~600 tokens).
        """
        model = self.llm_provider.get_chat_model()

        chain = self.skills_gen_prompt | model
        result = chain.invoke(
            {
                # Use compressed context instead of full JSON dump
                "reasoning_json": _compress_skills_reasoning(reasoning),
            }
        )
        return result.content

    # =========================================================================
    # CV ASSEMBLY
    # =========================================================================

    def assemble_cv(
        self,
        cv_data: CVData,
        tailored_summary: str,
        tailored_experiences: list[str],
        tailored_skills: str,
    ) -> str:
        """Assemble the final tailored CV.

        Args:
            cv_data: Original CV data.
            tailored_summary: Tailored summary.
            tailored_experiences: List of tailored experience texts.
            tailored_skills: Tailored skills section.

        Returns:
            str: Complete tailored CV in markdown format.
        """
        model = self.llm_provider.get_chat_model()

        # Format contact info - prefer raw_contact_line to preserve markdown links
        contact = cv_data.contact
        if contact.raw_contact_line:
            # Use exact original formatting with markdown links preserved
            contact_str = f"# {contact.name}\n{contact.raw_contact_line}"
        else:
            # Fallback to individual fields if raw line not available
            contact_str = f"**{contact.name}**\n"
            if contact.email:
                contact_str += f"Email: {contact.email}\n"
            if contact.phone:
                contact_str += f"Phone: {contact.phone}\n"
            if contact.location:
                contact_str += f"Location: {contact.location}\n"
            if contact.linkedin:
                contact_str += f"LinkedIn: {contact.linkedin}\n"
            if contact.github:
                contact_str += f"GitHub: {contact.github}\n"

        # Format education - prefer raw text to preserve EXACT original formatting
        # Education is IMMUTABLE during tailoring - never modify it
        if cv_data.raw_education_text:
            education_str = cv_data.raw_education_text
        else:
            # Fallback to structured data if raw text not available
            education_str = ""
            for edu in cv_data.education:
                education_str += f"**{edu.degree}** - {edu.institution} ({edu.graduation_date})\n"
                if edu.gpa:
                    education_str += f"GPA: {edu.gpa}\n"

        # Format certifications
        certs_str = ""
        for cert in cv_data.certifications:
            certs_str += f"- {cert.name} ({cert.issuer})\n"

        # Format publications
        pubs_str = ""
        for pub in cv_data.publications:
            pub_line = f"- {pub.title} ({pub.publisher}"
            if pub.year:
                pub_line += f", {pub.year}"
            pub_line += ")"
            if pub.url:
                pub_line += f": {pub.url}"
            pubs_str += pub_line + "\n"

        chain = self.assembly_prompt | model
        result = chain.invoke(
            {
                "contact": contact_str,
                "tailored_summary": tailored_summary,
                "tailored_experiences": "\n\n".join(tailored_experiences),
                "tailored_skills": tailored_skills,
                "education": education_str or "Not provided",
                "certifications": certs_str or "Not provided",
                "publications": pubs_str or "Not provided",
            }
        )

        return result.content
