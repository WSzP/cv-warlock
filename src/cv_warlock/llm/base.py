"""Base LLM provider abstraction."""

from abc import ABC, abstractmethod
from typing import Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_chat_model(self) -> BaseChatModel:
        """Get a chat model instance with API default temperature."""
        pass

    @abstractmethod
    def get_extraction_model(self) -> BaseChatModel:
        """Get a model optimized for structured extraction."""
        pass

    def extract_structured(
        self,
        prompt: str,
        output_schema: type[T],
        **kwargs: object,
    ) -> T:
        """Extract structured data using the model."""
        model = self.get_extraction_model()
        structured_model = model.with_structured_output(output_schema, method="function_calling")
        return structured_model.invoke(prompt, **kwargs)


def get_llm_provider(
    provider: Literal["openai", "anthropic", "google"],
    model: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """Factory function to get an LLM provider instance."""
    if provider == "openai":
        from cv_warlock.llm.openai import OpenAIProvider

        return OpenAIProvider(model=model or "gpt-5.2", api_key=api_key)
    elif provider == "anthropic":
        from cv_warlock.llm.anthropic import AnthropicProvider

        return AnthropicProvider(model=model or "claude-sonnet-4-5-20250929", api_key=api_key)
    elif provider == "google":
        from cv_warlock.llm.google import GoogleProvider

        return GoogleProvider(model=model or "gemini-3-flash-preview", api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")
