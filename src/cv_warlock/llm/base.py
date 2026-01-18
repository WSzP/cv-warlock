"""Base LLM provider abstraction."""

from abc import ABC, abstractmethod
from typing import Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Implements model caching to avoid repeated instantiation overhead.
    Models are cached on first access and reused for subsequent calls.
    """

    _chat_model: BaseChatModel | None = None
    _extraction_model: BaseChatModel | None = None

    def get_chat_model(self) -> BaseChatModel:
        """Get a cached chat model instance with API default temperature."""
        if self._chat_model is None:
            self._chat_model = self._create_chat_model()
        return self._chat_model

    def get_extraction_model(self) -> BaseChatModel:
        """Get a cached model optimized for structured extraction."""
        if self._extraction_model is None:
            self._extraction_model = self._create_extraction_model()
        return self._extraction_model

    @abstractmethod
    def _create_chat_model(self) -> BaseChatModel:
        """Create a new chat model instance. Override in subclasses."""
        pass

    @abstractmethod
    def _create_extraction_model(self) -> BaseChatModel:
        """Create a new extraction model instance. Override in subclasses."""
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
