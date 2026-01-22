"""RLM-aware LLM Provider.

Extends the base LLM provider to use RLM orchestration for large context handling.
Falls back to direct calls for small inputs.
"""

import logging
from typing import Any, Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from cv_warlock.llm.base import LLMProvider, get_llm_provider
from cv_warlock.rlm import RLMConfig, RLMOrchestrator, RLMResult

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class RLMProvider(LLMProvider):
    """LLM Provider that uses RLM for large context handling.

    This provider wraps a standard LLM provider and adds RLM orchestration
    capabilities for handling documents that exceed the size threshold.

    Features:
    - Automatic size-based routing (direct vs RLM)
    - Configurable root and sub-model selection
    - Graceful fallback on RLM failures
    - Full trajectory logging for observability
    """

    def __init__(
        self,
        root_provider: Literal["openai", "anthropic", "google"] = "anthropic",
        root_model: str = "claude-opus-4-5-20251101",
        sub_provider: Literal["openai", "anthropic", "google"] | None = None,
        sub_model: str | None = None,
        api_key: str | None = None,
        config: RLMConfig | None = None,
    ):
        """Initialize the RLM-aware provider.

        Args:
            root_provider: Provider for root model (needs strong coding ability).
            root_model: Model ID for root model.
            sub_provider: Provider for sub-calls (defaults to root provider).
            sub_model: Model ID for sub-calls (defaults to faster model).
            api_key: API key for the provider.
            config: RLM configuration options.
        """
        self.root_provider_name = root_provider
        self.root_model = root_model
        self.sub_provider_name = sub_provider or root_provider
        self.sub_model = sub_model
        self.api_key = api_key
        self.config = config or RLMConfig()

        # Default to faster model for sub-calls
        if self.sub_model is None:
            self.sub_model = self._get_default_sub_model()

        # Initialize underlying providers
        self._root_provider = get_llm_provider(root_provider, root_model, api_key)
        self._sub_provider = get_llm_provider(self.sub_provider_name, self.sub_model, api_key)

        # Initialize cache attributes from parent
        self._chat_model = None
        self._extraction_model = None

        # Create orchestrator lazily
        self._orchestrator: RLMOrchestrator | None = None

        # Last RLM result for observability
        self.last_rlm_result: RLMResult | None = None

    def _get_default_sub_model(self) -> str:
        """Get default model for sub-calls (Sonnet-tier for quality)."""
        if self.sub_provider_name == "anthropic":
            return "claude-sonnet-4-5-20250929"
        elif self.sub_provider_name == "openai":
            return "gpt-5.2"
        elif self.sub_provider_name == "google":
            return "gemini-3-pro-preview"
        return self.root_model

    def _get_orchestrator(self) -> RLMOrchestrator:
        """Get or create the RLM orchestrator."""
        if self._orchestrator is None:
            self._orchestrator = RLMOrchestrator(
                root_provider=self._root_provider,
                sub_provider=self._sub_provider,
                config=self.config,
            )
        return self._orchestrator

    def _create_chat_model(self) -> BaseChatModel:
        """Create chat model using root provider."""
        return self._root_provider.get_chat_model()

    def _create_extraction_model(self) -> BaseChatModel:
        """Create extraction model using root provider."""
        return self._root_provider.get_extraction_model()

    def should_use_rlm(self, cv_text: str, job_text: str) -> bool:
        """Determine if RLM should be used based on input size.

        Args:
            cv_text: CV text.
            job_text: Job specification text.

        Returns:
            True if RLM should be used.
        """
        total_size = len(cv_text) + len(job_text)
        return total_size > self.config.size_threshold

    def analyze_with_rlm(
        self,
        task: str,
        cv_text: str,
        job_text: str,
        output_schema: type[T] | None = None,
    ) -> T | Any:
        """Perform analysis using RLM orchestration.

        Args:
            task: The analysis task to perform.
            cv_text: CV text.
            job_text: Job specification text.
            output_schema: Optional Pydantic model for structured output.

        Returns:
            Analysis result (structured if schema provided).
        """
        orchestrator = self._get_orchestrator()

        logger.info(f"Using RLM for analysis (cv={len(cv_text)} chars, job={len(job_text)} chars)")

        result = orchestrator.complete(
            task=task,
            cv_text=cv_text,
            job_text=job_text,
            output_schema=output_schema,
        )

        # Store for observability
        self.last_rlm_result = result

        if not result.success:
            logger.warning(f"RLM analysis failed: {result.error}")
            raise RuntimeError(f"RLM analysis failed: {result.error}")

        logger.info(
            f"RLM complete: {result.total_iterations} iterations, "
            f"{result.sub_call_count} sub-calls, "
            f"{result.execution_time_seconds:.1f}s"
        )

        return result.answer

    def extract_structured_with_rlm(
        self,
        prompt: str,
        output_schema: type[T],
        cv_text: str,
        job_text: str,
    ) -> T:
        """Extract structured data using RLM.

        Args:
            prompt: Task description/prompt.
            output_schema: Pydantic model for output.
            cv_text: CV text.
            job_text: Job specification text.

        Returns:
            Structured extraction result.
        """
        return self.analyze_with_rlm(
            task=prompt,
            cv_text=cv_text,
            job_text=job_text,
            output_schema=output_schema,
        )

    def extract_with_fallback(
        self,
        prompt: str,
        output_schema: type[T],
        cv_text: str,
        job_text: str,
        force_rlm: bool = False,
    ) -> T:
        """Extract with automatic RLM/direct routing and fallback.

        Args:
            prompt: Extraction prompt.
            output_schema: Pydantic model for output.
            cv_text: CV text.
            job_text: Job specification text.
            force_rlm: If True, always use RLM.

        Returns:
            Extracted data.
        """
        use_rlm = force_rlm or self.should_use_rlm(cv_text, job_text)

        if use_rlm:
            try:
                return self.extract_structured_with_rlm(prompt, output_schema, cv_text, job_text)
            except Exception as e:
                logger.warning(f"RLM extraction failed, falling back to direct: {e}")
                # Fall through to direct extraction

        # Direct extraction
        full_prompt = f"{prompt}\n\nCV:\n{cv_text}\n\nJob:\n{job_text}"
        return self.extract_structured(full_prompt, output_schema)

    def get_trajectory_summary(self) -> dict[str, Any] | None:
        """Get summary of last RLM execution for observability.

        Returns:
            Dict with trajectory summary or None if no RLM run.
        """
        if self.last_rlm_result is None:
            return None

        result = self.last_rlm_result
        return {
            "success": result.success,
            "total_iterations": result.total_iterations,
            "sub_call_count": result.sub_call_count,
            "execution_time_seconds": result.execution_time_seconds,
            "error": result.error,
            "intermediate_findings_count": len(result.intermediate_findings),
            "trajectory_steps": len(result.trajectory),
        }


def get_rlm_provider(
    root_provider: Literal["openai", "anthropic", "google"] = "anthropic",
    root_model: str = "claude-opus-4-5-20251101",
    sub_provider: Literal["openai", "anthropic", "google"] | None = None,
    sub_model: str | None = None,
    api_key: str | None = None,
    config: RLMConfig | None = None,
) -> RLMProvider:
    """Factory function to create an RLM-aware provider.

    Args:
        root_provider: Provider for root model.
        root_model: Model ID for root model.
        sub_provider: Provider for sub-calls.
        sub_model: Model ID for sub-calls.
        api_key: API key.
        config: RLM configuration.

    Returns:
        Configured RLMProvider.
    """
    return RLMProvider(
        root_provider=root_provider,
        root_model=root_model,
        sub_provider=sub_provider,
        sub_model=sub_model,
        api_key=api_key,
        config=config,
    )
