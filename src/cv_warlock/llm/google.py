"""Google Gemini LLM provider."""

import os

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from cv_warlock.llm.base import LLMProvider


class GoogleProvider(LLMProvider):
    """Google Gemini provider using langchain-google-genai."""

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        api_key: str | None = None,
        temperature: float = 0.3,
    ):
        """Initialize the Google provider.

        Args:
            model: Model name (default: gemini-3-flash-preview).
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY env var.
            temperature: Default temperature for generation.
        """
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.default_temperature = temperature

    def get_chat_model(self, temperature: float | None = None) -> BaseChatModel:
        """Get a Gemini chat model instance."""
        return ChatGoogleGenerativeAI(
            model=self.model,
            temperature=temperature if temperature is not None else self.default_temperature,
            google_api_key=self.api_key,
        )

    def get_extraction_model(self) -> BaseChatModel:
        """Get a Gemini model optimized for structured extraction (temperature=0)."""
        return ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0,
            google_api_key=self.api_key,
        )
