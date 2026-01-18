"""Google Gemini LLM provider."""

import os

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from cv_warlock.llm.base import LLMProvider

# Default timeout in seconds for API requests
DEFAULT_TIMEOUT = 120.0  # 2 minutes per request
DEFAULT_MAX_RETRIES = 3  # 3 retries = 4 total attempts (handles transient connection errors)


class GoogleProvider(LLMProvider):
    """Google Gemini provider with model caching.

    Uses API default temperature (not passed explicitly).
    """

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """Initialize the Google provider.

        Args:
            model: Model name (default: gemini-3-flash-preview).
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY env var.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries on failure.
        """
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        # Initialize cache attributes from parent
        self._chat_model = None
        self._extraction_model = None

    def _create_chat_model(self) -> BaseChatModel:
        """Create a Gemini chat model with API default temperature."""
        return ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _create_extraction_model(self) -> BaseChatModel:
        """Create a Gemini model for structured extraction with API default temperature."""
        return ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
