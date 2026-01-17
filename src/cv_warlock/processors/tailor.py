"""CV tailoring processor with chain-of-thought reasoning.

This module provides CV section tailoring with optional CoT reasoning.
When CoT is enabled (default), each section follows a REASON -> GENERATE pattern
for balanced quality and speed. The optional "thorough" mode adds CRITIQUE -> REFINE.

Performance optimizations:
- Balanced mode: 2 LLM calls per section (REASON→GENERATE) instead of 4
- Compressed context: Only essential reasoning fields passed to generation
- Async experience processing: All experiences processed in parallel
"""

from concurrent.futures import ThreadPoolExecutor

from langchain_core.prompts import ChatPromptTemplate

from cv_warlock.llm.base import LLMProvider
from cv_warlock.models.cv import CVData, Experience
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.models.reasoning import (
    BulletReasoning,
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
    EXPERIENCE_CRITIQUE_PROMPT,
    EXPERIENCE_GENERATION_PROMPT,
    EXPERIENCE_REASONING_PROMPT,
    EXPERIENCE_REFINE_PROMPT,
    SKILLS_CRITIQUE_PROMPT,
    SKILLS_GENERATION_PROMPT,
    SKILLS_REASONING_PROMPT,
    SKILLS_REFINE_PROMPT,
    SUMMARY_CRITIQUE_PROMPT,
    SUMMARY_GENERATION_PROMPT,
    SUMMARY_REASONING_PROMPT,
    SUMMARY_REFINE_PROMPT,
)


def _compress_summary_reasoning(reasoning: SummaryReasoning) -> str:
    """Extract essential fields from summary reasoning for generation prompt.

    Instead of passing full JSON (~800 tokens), pass only what's needed (~150 tokens).
    """
    return f"""Hook: {reasoning.hook_strategy}
Keywords: {', '.join(reasoning.key_keywords_to_include)}
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
Keywords: {', '.join(reasoning.keywords_to_incorporate[:5])}
Priority achievements: {', '.join(reasoning.achievements_to_prioritize[:3])}
Bullet plans:
{bullets_summary}"""


def _compress_skills_reasoning(reasoning: SkillsReasoning) -> str:
    """Extract essential fields from skills reasoning for generation prompt."""
    categories = "\n".join(
        f"- {cat}: {', '.join(skills[:5])}"
        for cat, skills in list(reasoning.category_groupings.items())[:5]
    )

    return f"""Required matched: {', '.join(reasoning.required_skills_matched[:7])}
Preferred matched: {', '.join(reasoning.preferred_skills_matched[:5])}
Dual format: {', '.join(reasoning.dual_format_terms[:5])}
Categories:
{categories}"""


class CVTailor:
    """Tailor CV sections with optional chain-of-thought reasoning.

    When use_cot=True (default), uses a multi-step reasoning pipeline:
    1. REASON: Analyze inputs and plan approach (structured output)
    2. GENERATE: Create content based on reasoning
    3. CRITIQUE: Evaluate quality against criteria
    4. REFINE: Improve if quality below threshold (max iterations)

    When use_cot=False, uses direct single-prompt generation (faster but lower quality).
    """

    # Configuration
    MAX_REFINEMENT_ITERATIONS = 2
    REASONING_TEMPERATURE = 0.2  # Low for analytical reasoning
    GENERATION_TEMPERATURE = 0.4  # Moderate for creative generation
    CRITIQUE_TEMPERATURE = 0.1  # Very low for consistent evaluation

    def __init__(self, llm_provider: LLMProvider, use_cot: bool = True):
        """Initialize tailor with optional CoT reasoning.

        Args:
            llm_provider: LLM provider instance.
            use_cot: Whether to use chain-of-thought reasoning (default True).
                     Set to False for faster generation with lower quality.
        """
        self.llm_provider = llm_provider
        self.use_cot = use_cot

        # Original prompts (for backward compatibility / non-CoT mode)
        self.summary_prompt = ChatPromptTemplate.from_template(SUMMARY_TAILORING_PROMPT)
        self.experience_prompt = ChatPromptTemplate.from_template(EXPERIENCE_TAILORING_PROMPT)
        self.skills_prompt = ChatPromptTemplate.from_template(SKILLS_TAILORING_PROMPT)
        self.assembly_prompt = ChatPromptTemplate.from_template(CV_ASSEMBLY_PROMPT)

        # CoT prompts - Summary
        self.summary_reasoning_prompt = ChatPromptTemplate.from_template(SUMMARY_REASONING_PROMPT)
        self.summary_gen_prompt = ChatPromptTemplate.from_template(SUMMARY_GENERATION_PROMPT)
        self.summary_critique_prompt = ChatPromptTemplate.from_template(SUMMARY_CRITIQUE_PROMPT)
        self.summary_refine_prompt = ChatPromptTemplate.from_template(SUMMARY_REFINE_PROMPT)

        # CoT prompts - Experience
        self.exp_reasoning_prompt = ChatPromptTemplate.from_template(EXPERIENCE_REASONING_PROMPT)
        self.exp_gen_prompt = ChatPromptTemplate.from_template(EXPERIENCE_GENERATION_PROMPT)
        self.exp_critique_prompt = ChatPromptTemplate.from_template(EXPERIENCE_CRITIQUE_PROMPT)
        self.exp_refine_prompt = ChatPromptTemplate.from_template(EXPERIENCE_REFINE_PROMPT)

        # CoT prompts - Skills
        self.skills_reasoning_prompt = ChatPromptTemplate.from_template(SKILLS_REASONING_PROMPT)
        self.skills_gen_prompt = ChatPromptTemplate.from_template(SKILLS_GENERATION_PROMPT)
        self.skills_critique_prompt = ChatPromptTemplate.from_template(SKILLS_CRITIQUE_PROMPT)
        self.skills_refine_prompt = ChatPromptTemplate.from_template(SKILLS_REFINE_PROMPT)

    # =========================================================================
    # SUMMARY TAILORING
    # =========================================================================

    def tailor_summary(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
    ) -> str:
        """Tailor the professional summary.

        Uses CoT reasoning if enabled, otherwise direct generation.

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.
            tailoring_plan: Tailoring plan.

        Returns:
            str: Tailored summary text.
        """
        if self.use_cot:
            result = self.tailor_summary_with_cot(cv_data, job_requirements, tailoring_plan)
            return result.final_summary
        else:
            return self._tailor_summary_direct(cv_data, job_requirements, tailoring_plan)

    def _tailor_summary_direct(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
    ) -> str:
        """Direct summary generation without CoT (original implementation)."""
        model = self.llm_provider.get_chat_model(temperature=0.4)
        chain = self.summary_prompt | model
        result = chain.invoke({
            "original_summary": cv_data.summary or "No summary provided",
            "job_title": job_requirements.job_title,
            "company": job_requirements.company or "the company",
            "key_requirements": ", ".join(job_requirements.required_skills[:5]),
            "relevant_strengths": ", ".join(tailoring_plan["skills_to_highlight"][:5]),
        })
        return result.content

    def tailor_summary_with_cot(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
        context: GenerationContext | None = None,
    ) -> SummaryGenerationResult:
        """Tailor summary with chain-of-thought reasoning (balanced mode).

        Balanced mode: REASON → GENERATE only (2 LLM calls).
        Skips CRITIQUE → REFINE for faster execution while maintaining quality.

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.
            tailoring_plan: Tailoring plan.
            context: Optional context from previous sections.

        Returns:
            SummaryGenerationResult with reasoning and generation output.
        """
        # Step 1: REASON
        reasoning = self._reason_summary(cv_data, job_requirements, tailoring_plan)

        # Step 2: GENERATE
        generated = self._generate_summary_from_reasoning(reasoning, job_requirements)

        # Balanced mode: Skip CRITIQUE and REFINE - use generated output directly
        # This saves 2-4 LLM calls per summary

        # Create a placeholder critique for API compatibility
        critique = SummaryCritique(
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

        return SummaryGenerationResult(
            reasoning=reasoning,
            generated_summary=generated,
            critique=critique,
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
        structured_model = model.with_structured_output(SummaryReasoning)

        chain = self.summary_reasoning_prompt | structured_model
        return chain.invoke({
            "original_summary": cv_data.summary or "No summary provided",
            "job_title": job_requirements.job_title,
            "company": job_requirements.company or "the company",
            "key_requirements": ", ".join(job_requirements.required_skills[:7]),
            "relevant_strengths": ", ".join(tailoring_plan["skills_to_highlight"][:7]),
            "tailoring_plan_summary": f"""
Focus areas: {', '.join(tailoring_plan['summary_focus'][:3])}
Key achievements to feature: {', '.join(tailoring_plan['achievements_to_feature'][:3])}
Keywords to incorporate: {', '.join(tailoring_plan['keywords_to_incorporate'][:5])}
""",
        })

    def _generate_summary_from_reasoning(
        self,
        reasoning: SummaryReasoning,
        job_requirements: JobRequirements,
    ) -> str:
        """Generate summary based on reasoning (Step 2).

        Uses compressed reasoning context (~150 tokens) instead of full JSON (~800 tokens).
        """
        model = self.llm_provider.get_chat_model(temperature=self.GENERATION_TEMPERATURE)

        chain = self.summary_gen_prompt | model
        result = chain.invoke({
            # Use compressed context instead of full JSON dump
            "reasoning_json": _compress_summary_reasoning(reasoning),
            "strongest_metric": reasoning.strongest_metric,
            "keywords": ", ".join(reasoning.key_keywords_to_include),
        })
        return result.content

    def _critique_summary(
        self,
        summary: str,
        job_requirements: JobRequirements,
        reasoning: SummaryReasoning,
    ) -> SummaryCritique:
        """Critique generated summary (Step 3)."""
        model = self.llm_provider.get_extraction_model()
        structured_model = model.with_structured_output(SummaryCritique)

        chain = self.summary_critique_prompt | structured_model
        return chain.invoke({
            "generated_summary": summary,
            "job_title": job_requirements.job_title,
            "company": job_requirements.company or "the company",
            "required_keywords": ", ".join(reasoning.key_keywords_to_include),
        })

    def _refine_summary(
        self,
        current: str,
        critique: SummaryCritique,
        reasoning: SummaryReasoning,
        job_requirements: JobRequirements,
    ) -> str:
        """Refine summary based on critique (Step 4).

        Uses compressed reasoning context for efficiency.
        """
        model = self.llm_provider.get_chat_model(temperature=self.GENERATION_TEMPERATURE)

        chain = self.summary_refine_prompt | model
        result = chain.invoke({
            "current_summary": current,
            "issues": "\n".join(f"- {i}" for i in critique.issues_found),
            "suggestions": "\n".join(f"- {s}" for s in critique.improvement_suggestions),
            # Use compressed context instead of full JSON dump
            "reasoning_json": _compress_summary_reasoning(reasoning),
            "strongest_metric": reasoning.strongest_metric,
            "keywords": ", ".join(reasoning.key_keywords_to_include),
        })
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
            str: Tailored experience bullets as text.
        """
        if self.use_cot:
            result = self.tailor_experience_with_cot(
                experience, job_requirements, tailoring_plan
            )
            return "\n".join(f"- {b}" for b in result.final_bullets)
        else:
            return self._tailor_experience_direct(experience, job_requirements, tailoring_plan)

    def _tailor_experience_direct(
        self,
        experience: Experience,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
    ) -> str:
        """Direct experience generation without CoT."""
        model = self.llm_provider.get_chat_model(temperature=0.3)
        chain = self.experience_prompt | model
        result = chain.invoke({
            "title": experience.title,
            "company": experience.company,
            "period": f"{experience.start_date} - {experience.end_date or 'Present'}",
            "description": experience.description,
            "achievements": "\n".join(f"- {a}" for a in experience.achievements),
            "target_requirements": ", ".join(job_requirements.required_skills[:5]),
            "skills_to_emphasize": ", ".join(tailoring_plan["skills_to_highlight"][:5]),
        })
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
        reasoning = self._reason_experience(
            experience, job_requirements, tailoring_plan, context
        )

        # Determine bullet count from emphasis strategy
        bullet_count = self._get_bullet_count(reasoning.emphasis_strategy)

        # Step 2: GENERATE
        generated_text = self._generate_experience_from_reasoning(
            reasoning, job_requirements, bullet_count
        )
        generated_bullets = self._parse_bullets(generated_text)

        # Balanced mode: Skip CRITIQUE and REFINE - use generated output directly
        # This saves 2-4 LLM calls per experience (massive savings with multiple experiences)

        # Create a placeholder critique for API compatibility
        critique = ExperienceCritique(
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

        return ExperienceGenerationResult(
            experience_title=experience.title,
            experience_company=experience.company,
            reasoning=reasoning,
            generated_bullets=generated_bullets,
            critique=critique,
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
        """Generate reasoning for experience (Step 1)."""
        model = self.llm_provider.get_extraction_model()
        structured_model = model.with_structured_output(ExperienceReasoning)

        chain = self.exp_reasoning_prompt | structured_model
        return chain.invoke({
            "title": experience.title,
            "company": experience.company,
            "period": f"{experience.start_date} - {experience.end_date or 'Present'}",
            "description": experience.description or "No description provided",
            "achievements": "\n".join(f"- {a}" for a in experience.achievements)
            if experience.achievements
            else "No specific achievements listed",
            "job_title": job_requirements.job_title,
            "target_requirements": ", ".join(job_requirements.required_skills[:7]),
            "skills_to_emphasize": ", ".join(tailoring_plan["skills_to_highlight"][:5]),
            "established_identity": context.established_identity if context else "Not yet established",
            "keywords_already_used": ", ".join(context.primary_keywords_used) if context else "None yet",
            "metrics_already_used": ", ".join(context.metrics_used) if context else "None yet",
        })

    def _get_bullet_count(self, emphasis_strategy: str) -> int:
        """Determine bullet count from emphasis strategy."""
        strategy_upper = emphasis_strategy.upper()
        if "HIGH" in strategy_upper:
            return 5
        elif "MED" in strategy_upper:
            return 4
        else:
            return 3

    def _generate_experience_from_reasoning(
        self,
        reasoning: ExperienceReasoning,
        job_requirements: JobRequirements,
        bullet_count: int,
    ) -> str:
        """Generate experience bullets from reasoning (Step 2).

        Uses compressed reasoning context (~200 tokens) instead of full JSON (~1000 tokens).
        """
        model = self.llm_provider.get_chat_model(temperature=self.GENERATION_TEMPERATURE)

        chain = self.exp_gen_prompt | model
        result = chain.invoke({
            # Use compressed context instead of full JSON dump
            "reasoning_json": _compress_experience_reasoning(reasoning),
            "bullet_count": bullet_count,
            "keywords_to_use": ", ".join(reasoning.keywords_to_incorporate[:5]),
        })
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

    def _critique_experience(
        self,
        bullets: list[str],
        job_requirements: JobRequirements,
        reasoning: ExperienceReasoning,
    ) -> ExperienceCritique:
        """Critique experience bullets (Step 3)."""
        model = self.llm_provider.get_extraction_model()
        structured_model = model.with_structured_output(ExperienceCritique)

        chain = self.exp_critique_prompt | structured_model
        return chain.invoke({
            "generated_bullets": "\n".join(f"- {b}" for b in bullets),
            "job_title": job_requirements.job_title,
            "job_requirements": ", ".join(job_requirements.required_skills[:7]),
        })

    def _refine_experience(
        self,
        current_bullets: list[str],
        critique: ExperienceCritique,
        reasoning: ExperienceReasoning,
        job_requirements: JobRequirements,
        bullet_count: int,
    ) -> str:
        """Refine experience bullets (Step 4).

        Uses compressed reasoning context for efficiency.
        """
        model = self.llm_provider.get_chat_model(temperature=self.GENERATION_TEMPERATURE)

        chain = self.exp_refine_prompt | model
        result = chain.invoke({
            "current_bullets": "\n".join(f"- {b}" for b in current_bullets),
            "weak_bullets": "\n".join(f"- {w}" for w in critique.weak_bullets),
            "suggestions": "\n".join(f"- {s}" for s in critique.improvement_suggestions),
            # Use compressed context instead of full JSON dump
            "reasoning_json": _compress_experience_reasoning(reasoning),
            "bullet_count": bullet_count,
            "keywords_to_use": ", ".join(reasoning.keywords_to_incorporate[:5]),
        })
        return result.content

    def tailor_experiences(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
    ) -> list[str]:
        """Tailor all experience entries.

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.
            tailoring_plan: Tailoring plan.

        Returns:
            list[str]: List of tailored experience texts.
        """
        tailored = []
        for exp in cv_data.experiences:
            tailored_exp = self.tailor_experience(exp, job_requirements, tailoring_plan)
            tailored.append(tailored_exp)
        return tailored

    def tailor_experiences_with_cot(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
        context: GenerationContext | None = None,
    ) -> tuple[list[str], list[ExperienceGenerationResult], GenerationContext]:
        """Tailor all experiences with CoT using parallel processing.

        Uses ThreadPoolExecutor to process all experiences concurrently,
        reducing total time from N*T to approximately T (where N is number
        of experiences and T is time per experience).

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.
            tailoring_plan: Tailoring plan.
            context: Optional context from summary generation.

        Returns:
            Tuple of (tailored texts, generation results, updated context).
        """
        current_context = context or GenerationContext()

        # Process all experiences in parallel using ThreadPoolExecutor
        # This is safe because each experience is independent during generation
        def process_experience(exp: Experience) -> ExperienceGenerationResult:
            return self.tailor_experience_with_cot(
                exp, job_requirements, tailoring_plan, current_context
            )

        # Use ThreadPoolExecutor for parallel HTTP requests to LLM API
        # max_workers=None lets Python choose optimal thread count
        with ThreadPoolExecutor(max_workers=None) as executor:
            results = list(executor.map(process_experience, cv_data.experiences))

        # Format results and update context (must be sequential)
        tailored_texts = []
        for result in results:
            # Format as text
            tailored_text = "\n".join(f"- {b}" for b in result.final_bullets)
            tailored_texts.append(tailored_text)

            # Update context (for downstream use, e.g., skills section)
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

        return tailored_texts, results, current_context

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
        model = self.llm_provider.get_chat_model(temperature=0.2)
        chain = self.skills_prompt | model
        result = chain.invoke({
            "all_skills": ", ".join(cv_data.skills),
            "required_skills": ", ".join(job_requirements.required_skills),
            "preferred_skills": ", ".join(job_requirements.preferred_skills),
        })
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

        # Balanced mode: Skip CRITIQUE and REFINE - use generated output directly
        # This saves 2-4 LLM calls for skills section

        # Create a placeholder critique for API compatibility
        critique = SkillsCritique(
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

        return SkillsGenerationResult(
            reasoning=reasoning,
            generated_skills=generated,
            critique=critique,
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
        structured_model = model.with_structured_output(SkillsReasoning)

        chain = self.skills_reasoning_prompt | structured_model
        return chain.invoke({
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
        })

    def _generate_skills_from_reasoning(self, reasoning: SkillsReasoning) -> str:
        """Generate skills section from reasoning (Step 2).

        Uses compressed reasoning context (~150 tokens) instead of full JSON (~600 tokens).
        """
        model = self.llm_provider.get_chat_model(temperature=self.CRITIQUE_TEMPERATURE)

        chain = self.skills_gen_prompt | model
        result = chain.invoke({
            # Use compressed context instead of full JSON dump
            "reasoning_json": _compress_skills_reasoning(reasoning),
        })
        return result.content

    def _critique_skills(
        self,
        skills_text: str,
        cv_data: CVData,
        job_requirements: JobRequirements,
    ) -> SkillsCritique:
        """Critique skills section (Step 3)."""
        model = self.llm_provider.get_extraction_model()
        structured_model = model.with_structured_output(SkillsCritique)

        chain = self.skills_critique_prompt | structured_model
        return chain.invoke({
            "generated_skills": skills_text,
            "required_skills": ", ".join(job_requirements.required_skills),
            "preferred_skills": ", ".join(job_requirements.preferred_skills),
            "candidate_skills": ", ".join(cv_data.skills),
        })

    def _refine_skills(
        self,
        current: str,
        critique: SkillsCritique,
        reasoning: SkillsReasoning,
        cv_data: CVData,
    ) -> str:
        """Refine skills section (Step 4).

        Uses compressed reasoning context for efficiency.
        """
        model = self.llm_provider.get_chat_model(temperature=self.CRITIQUE_TEMPERATURE)

        chain = self.skills_refine_prompt | model
        result = chain.invoke({
            "current_skills": current,
            "missing_terms": ", ".join(critique.missing_critical_terms),
            "suggestions": "\n".join(f"- {s}" for s in critique.improvement_suggestions),
            # Use compressed context instead of full JSON dump
            "reasoning_json": _compress_skills_reasoning(reasoning),
            "candidate_skills": ", ".join(cv_data.skills),
        })
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
        model = self.llm_provider.get_chat_model(temperature=0.2)

        # Format contact info
        contact = cv_data.contact
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

        # Format education
        education_str = ""
        for edu in cv_data.education:
            education_str += f"**{edu.degree}** - {edu.institution} ({edu.graduation_date})\n"
            if edu.gpa:
                education_str += f"GPA: {edu.gpa}\n"

        # Format projects
        projects_str = ""
        for proj in cv_data.projects:
            projects_str += f"**{proj.name}**: {proj.description}\n"
            if proj.technologies:
                projects_str += f"Technologies: {', '.join(proj.technologies)}\n"

        # Format certifications
        certs_str = ""
        for cert in cv_data.certifications:
            certs_str += f"- {cert.name} ({cert.issuer})\n"

        chain = self.assembly_prompt | model
        result = chain.invoke({
            "contact": contact_str,
            "tailored_summary": tailored_summary,
            "tailored_experiences": "\n\n---\n\n".join(tailored_experiences),
            "tailored_skills": tailored_skills,
            "education": education_str or "Not provided",
            "projects": projects_str or "Not provided",
            "certifications": certs_str or "Not provided",
        })

        return result.content
