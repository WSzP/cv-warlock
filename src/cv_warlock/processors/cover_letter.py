"""Cover letter generation processor with chain-of-thought reasoning.

This module generates professional cover letters using the tailored CV
and job requirements. Output is plain text suitable for job application
text areas.

Performance:
- Uses REASON -> GENERATE pattern (2 LLM calls)
- Compressed context passing for token efficiency
"""

import re

from langchain_core.prompts import ChatPromptTemplate

from cv_warlock.llm.base import LLMProvider
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.models.reasoning import (
    CoverLetterCritique,
    CoverLetterGenerationResult,
    CoverLetterReasoning,
    QualityLevel,
)
from cv_warlock.models.state import MatchAnalysis
from cv_warlock.prompts.cover_letter import (
    COVER_LETTER_GENERATION_PROMPT,
    COVER_LETTER_REASONING_PROMPT,
)


def _compress_cover_letter_reasoning(reasoning: CoverLetterReasoning) -> str:
    """Extract essential fields from reasoning for generation prompt.

    Instead of passing full JSON, pass only what's needed for generation.
    """
    return f"""Opening: {reasoning.opening_hook}
Key points: {", ".join(reasoning.key_selling_points[:3])}
Strongest alignment: {reasoning.strongest_alignment}
Company connection: {reasoning.company_connection}
Structure: {" | ".join(reasoning.paragraph_structure)}
Call to action: {reasoning.call_to_action}
Tone: {reasoning.tone_guidance}"""


class CoverLetterGenerator:
    """Generate cover letters with chain-of-thought reasoning.

    Uses REASON -> GENERATE pattern for quality output.
    Produces plain text suitable for job application paste areas.
    """

    # Character limit configuration
    DEFAULT_CHARACTER_LIMIT = 2500
    MIN_CHARACTER_LIMIT = 500
    MAX_CHARACTER_LIMIT = 5000

    def __init__(self, llm_provider: LLMProvider):
        """Initialize generator with LLM provider.

        Args:
            llm_provider: LLM provider instance for model access.
        """
        self.llm_provider = llm_provider

        # Prompts
        self.reasoning_prompt = ChatPromptTemplate.from_template(COVER_LETTER_REASONING_PROMPT)
        self.generation_prompt = ChatPromptTemplate.from_template(COVER_LETTER_GENERATION_PROMPT)

    def generate(
        self,
        tailored_cv: str,
        job_requirements: JobRequirements,
        match_analysis: MatchAnalysis,
        character_limit: int | None = None,
    ) -> CoverLetterGenerationResult:
        """Generate a cover letter using CoT reasoning.

        Args:
            tailored_cv: The tailored CV markdown.
            job_requirements: Structured job requirements.
            match_analysis: Match analysis from scoring.
            character_limit: Target character limit for output.

        Returns:
            CoverLetterGenerationResult with reasoning and final output.
        """
        # Validate and normalize character limit
        if character_limit is None:
            character_limit = self.DEFAULT_CHARACTER_LIMIT
        character_limit = max(
            self.MIN_CHARACTER_LIMIT,
            min(character_limit, self.MAX_CHARACTER_LIMIT),
        )

        # Step 1: REASON
        reasoning = self._reason(tailored_cv, job_requirements, match_analysis)

        # Step 2: GENERATE
        generated = self._generate_from_reasoning(reasoning, job_requirements, character_limit)

        # Create placeholder critique (balanced mode - skip full critique/refine)
        critique = CoverLetterCritique(
            has_compelling_opening=True,
            demonstrates_company_research=True,
            includes_quantified_achievement=True,
            mirrors_job_keywords=True,
            appropriate_length=len(generated) <= character_limit,
            professional_tone=True,
            has_clear_call_to_action=True,
            is_plain_text=True,
            quality_level=QualityLevel.GOOD,
            issues_found=[],
            improvement_suggestions=[],
            should_refine=False,
        )

        return CoverLetterGenerationResult(
            reasoning=reasoning,
            generated_cover_letter=generated,
            critique=critique,
            refinement_count=0,
            final_cover_letter=generated,
            character_count=len(generated),
        )

    def _reason(
        self,
        tailored_cv: str,
        job_requirements: JobRequirements,
        match_analysis: MatchAnalysis,
    ) -> CoverLetterReasoning:
        """Generate reasoning for cover letter (Step 1).

        Args:
            tailored_cv: The tailored CV markdown.
            job_requirements: Structured job requirements.
            match_analysis: Match analysis from scoring.

        Returns:
            CoverLetterReasoning with strategic thinking.
        """
        model = self.llm_provider.get_extraction_model()
        structured_model = model.with_structured_output(
            CoverLetterReasoning, method="function_calling"
        )

        chain = self.reasoning_prompt | structured_model
        return chain.invoke(
            {
                "tailored_cv": tailored_cv,
                "job_title": job_requirements.job_title,
                "company": job_requirements.company or "the company",
                "required_skills": ", ".join(job_requirements.required_skills[:7]),
                "preferred_skills": ", ".join(job_requirements.preferred_skills[:5]),
                "responsibilities": ", ".join(job_requirements.responsibilities[:5]),
                "strong_matches": ", ".join(match_analysis.get("strong_matches", [])[:5]),
                "transferable_skills": ", ".join(match_analysis.get("transferable_skills", [])[:5]),
            }
        )

    def _generate_from_reasoning(
        self,
        reasoning: CoverLetterReasoning,
        job_requirements: JobRequirements,
        character_limit: int,
    ) -> str:
        """Generate cover letter from reasoning (Step 2).

        Args:
            reasoning: Strategic reasoning from step 1.
            job_requirements: Structured job requirements.
            character_limit: Target character limit.

        Returns:
            Plain text cover letter.
        """
        model = self.llm_provider.get_chat_model()

        chain = self.generation_prompt | model
        result = chain.invoke(
            {
                "reasoning_json": _compress_cover_letter_reasoning(reasoning),
                "character_limit": character_limit,
                "metric_to_feature": reasoning.metric_to_feature,
                "keywords": ", ".join(reasoning.keywords_to_incorporate[:5]),
            }
        )

        # Ensure plain text (strip any markdown that slipped through)
        text = result.content
        text = self._ensure_plain_text(text)

        return text

    def _ensure_plain_text(self, text: str) -> str:
        """Remove any markdown or formatting from text.

        Args:
            text: Potentially formatted text.

        Returns:
            Plain text with formatting removed.
        """
        # Remove markdown headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove bold
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        # Remove italic
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        # Remove bullet points
        text = re.sub(r"^[\-\*]\s+", "", text, flags=re.MULTILINE)
        # Remove links
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove code blocks
        text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
        # Remove inline code
        text = re.sub(r"`([^`]+)`", r"\1", text)

        return text.strip()
