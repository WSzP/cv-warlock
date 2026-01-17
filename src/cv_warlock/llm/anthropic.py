"""Anthropic Claude LLM provider."""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from cv_warlock.llm.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self,
        model: str = "claude-opus-4-5-20251101",
        api_key: str | None = None,
        temperature: float = 0.3,
    ):
        self.model = model
        self.api_key = api_key
        self.default_temperature = temperature

    def get_chat_model(self, temperature: float | None = None) -> BaseChatModel:
        """Get Anthropic chat model."""
        return ChatAnthropic(
            model=self.model,
            temperature=temperature if temperature is not None else self.default_temperature,
            api_key=self.api_key,
        )

    def get_extraction_model(self) -> BaseChatModel:
        """Get model for structured extraction with temperature=0."""
        return ChatAnthropic(
            model=self.model,
            temperature=0,
            api_key=self.api_key,
        )
