"""OpenAI LLM provider."""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from cv_warlock.llm.base import LLMProvider

# Default timeout in seconds for API requests
DEFAULT_TIMEOUT = 120.0  # 2 minutes per request
DEFAULT_MAX_RETRIES = 3  # 3 retries = 4 total attempts (handles transient connection errors)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider.

    Uses API default temperature (not passed explicitly).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def get_chat_model(self) -> BaseChatModel:
        """Get OpenAI chat model with API default temperature."""
        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def get_extraction_model(self) -> BaseChatModel:
        """Get model for structured extraction with API default temperature."""
        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
