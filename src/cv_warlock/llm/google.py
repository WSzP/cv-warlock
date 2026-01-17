"""Google Gemini LLM provider."""

import os

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from cv_warlock.llm.base import LLMProvider

# Default timeout in seconds for API requests
DEFAULT_TIMEOUT = 120.0  # 2 minutes per request
DEFAULT_MAX_RETRIES = 3  # 3 retries = 4 total attempts (handles transient connection errors)


class GoogleProvider(LLMProvider):
    """Google Gemini provider using langchain-google-genai."""

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        api_key: str | None = None,
        temperature: float = 0.3,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """Initialize the Google provider.

        Args:
            model: Model name (default: gemini-3-flash-preview).
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY env var.
            temperature: Default temperature for generation.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries on failure.
        """
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.default_temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def get_chat_model(self, temperature: float | None = None) -> BaseChatModel:
        """Get a Gemini chat model instance."""
        return ChatGoogleGenerativeAI(
            model=self.model,
            temperature=temperature if temperature is not None else self.default_temperature,
            google_api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def get_extraction_model(self) -> BaseChatModel:
        """Get a Gemini model optimized for structured extraction (temperature=0)."""
        return ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0,
            google_api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
