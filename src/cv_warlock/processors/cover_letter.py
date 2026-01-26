"""Cover letter generation processor with chain-of-thought reasoning.

This module generates professional cover letters using the tailored CV
and job requirements. Output is plain text suitable for job application
text areas.

Performance:
- Uses REASON -> GENERATE pattern (2 LLM calls)
- Uses same high-quality model for both steps (Sonnet/GPT-5.2) for
  professional wording quality (fast models produce lower quality text)
- Compressed context passing for token efficiency
- Optional fast_provider parameter for environments where speed is prioritized
"""

import re

from langchain_core.prompts import ChatPromptTemplate

from cv_warlock.llm.base import LLMProvider
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.models.reasoning import (
    CoverLetterGenerationResult,
    CoverLetterReasoning,
)
from cv_warlock.models.state import MatchAnalysis
from cv_warlock.prompts.cover_letter import (
    COVER_LETTER_REASONING_PROMPT,
    GENERATION_PROMPTS_BY_TIER,
)


def _compress_cover_letter_reasoning(reasoning: CoverLetterReasoning) -> str:
    """Compress reasoning to minimal tokens needed for generation.

    Only includes actionable guidance, not analysis.
    """
    return f"""Hook: {reasoning.opening_hook}
Points: {", ".join(reasoning.key_selling_points[:3])}
Match: {reasoning.strongest_alignment}
Company: {reasoning.company_connection}
Qualities: {", ".join(reasoning.leadership_qualities[:3])}
Problem-solution: {reasoning.problem_solution_framing}
Structure: {" > ".join(reasoning.paragraph_structure[:4])}
CTA: {reasoning.call_to_action}"""


class CoverLetterGenerator:
    """Generate cover letters with chain-of-thought reasoning.

    Uses REASON -> GENERATE pattern for quality output:
    - REASON: Uses extraction model (stronger, for structured output)
    - GENERATE: Uses fast model if provided (faster, plain text output)

    Produces plain text suitable for job application paste areas.
    """

    # Character limit configuration
    DEFAULT_CHARACTER_LIMIT = 2000
    MIN_CHARACTER_LIMIT = 400
    MAX_CHARACTER_LIMIT = 5000

    # Length tiers for structure adaptation
    TIER_MICRO = 600  # 400-600: 1-2 sentences, single point
    TIER_SHORT = 800  # 601-800: 2-3 sentences, 1-2 points
    TIER_COMPACT = 1200  # 801-1200: 2 paragraphs, 2-3 points
    TIER_STANDARD = 2000  # 1201+: 3-4 paragraphs, full structure

    def __init__(self, llm_provider: LLMProvider, fast_provider: LLMProvider | None = None):
        """Initialize generator with LLM providers.

        Args:
            llm_provider: LLM provider for reasoning (structured extraction).
            fast_provider: Optional fast provider for generation (plain text).
                          If not provided, uses llm_provider for both steps.
        """
        self.llm_provider = llm_provider
        self.fast_provider = fast_provider or llm_provider

        # Prompts
        self.reasoning_prompt = ChatPromptTemplate.from_template(COVER_LETTER_REASONING_PROMPT)
        # Generation prompts are selected dynamically based on character limit

    def _get_length_tier(self, character_limit: int) -> str:
        """Determine the length tier for structure adaptation.

        Args:
            character_limit: Target character limit.

        Returns:
            Tier name: 'micro', 'short', 'compact', or 'standard'.
        """
        if character_limit <= self.TIER_MICRO:
            return "micro"
        elif character_limit <= self.TIER_SHORT:
            return "short"
        elif character_limit <= self.TIER_COMPACT:
            return "compact"
        else:
            return "standard"

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

        # Step 1: REASON (uses extraction model for structured output)
        reasoning = self._reason(tailored_cv, job_requirements, match_analysis)

        # Step 2: GENERATE (uses fast model for plain text output)
        generated = self._generate_from_reasoning(reasoning, character_limit)

        return CoverLetterGenerationResult(
            reasoning=reasoning,
            generated_cover_letter=generated,
            critique=None,
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
        character_limit: int,
    ) -> str:
        """Generate cover letter from reasoning (Step 2).

        Uses the fast provider for generation since this is plain text output
        and doesn't require structured extraction. Selects the appropriate
        prompt template based on the character limit tier.

        Args:
            reasoning: Strategic reasoning from step 1.
            character_limit: Target character limit.

        Returns:
            Plain text cover letter.
        """
        # Select the appropriate prompt based on character limit
        tier = self._get_length_tier(character_limit)
        prompt_template = GENERATION_PROMPTS_BY_TIER[tier]
        generation_prompt = ChatPromptTemplate.from_template(prompt_template)

        # Use fast provider for generation (plain text, no structured output needed)
        model = self.fast_provider.get_chat_model()

        chain = generation_prompt | model
        result = chain.invoke(
            {
                "reasoning_json": _compress_cover_letter_reasoning(reasoning),
                "character_limit": character_limit,
                "metric_to_feature": reasoning.metric_to_feature,
                "keywords": ", ".join(reasoning.keywords_to_incorporate[:5]),
                "leadership_qualities": ", ".join(reasoning.leadership_qualities[:3]),
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
