"""Base LLM provider abstraction."""

from abc import ABC, abstractmethod
from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_chat_model(self, temperature: float | None = None) -> BaseChatModel:
        """Get a chat model instance.

        Args:
            temperature: Override temperature. If None, uses provider's default.
        """
        pass

    @abstractmethod
    def get_extraction_model(self) -> BaseChatModel:
        """Get a model optimized for structured extraction (temperature=0)."""
        pass

    def extract_structured[T: BaseModel](
        self,
        prompt: str,
        output_schema: type[T],
        **kwargs: object,
    ) -> T:
        """Extract structured data using the model."""
        model = self.get_extraction_model()
        structured_model = model.with_structured_output(output_schema)
        return structured_model.invoke(prompt, **kwargs)


def get_llm_provider(
    provider: Literal["openai", "anthropic", "google"],
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.3,
) -> LLMProvider:
    """Factory function to get an LLM provider instance."""
    if provider == "openai":
        from cv_warlock.llm.openai import OpenAIProvider

        return OpenAIProvider(model=model or "gpt-5.2", api_key=api_key, temperature=temperature)
    elif provider == "anthropic":
        from cv_warlock.llm.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=model or "claude-sonnet-4-5-20250929", api_key=api_key, temperature=temperature
        )
    elif provider == "google":
        from cv_warlock.llm.google import GoogleProvider

        return GoogleProvider(
            model=model or "gemini-3-flash-preview", api_key=api_key, temperature=temperature
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
