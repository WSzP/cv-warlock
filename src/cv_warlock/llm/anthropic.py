"""Anthropic Claude LLM provider."""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from cv_warlock.llm.base import LLMProvider

# Default timeout in seconds for API requests
# Set conservative timeout to prevent hanging on slow responses
DEFAULT_TIMEOUT = 120.0  # 2 minutes per request
DEFAULT_MAX_RETRIES = 1  # 1 retry = 2 total attempts max


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self,
        model: str = "claude-opus-4-5-20251101",
        api_key: str | None = None,
        temperature: float = 0.3,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.model = model
        self.api_key = api_key
        self.default_temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def get_chat_model(self, temperature: float | None = None) -> BaseChatModel:
        """Get Anthropic chat model."""
        return ChatAnthropic(
            model=self.model,
            temperature=temperature if temperature is not None else self.default_temperature,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def get_extraction_model(self) -> BaseChatModel:
        """Get model for structured extraction with temperature=0."""
        return ChatAnthropic(
            model=self.model,
            temperature=0,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
