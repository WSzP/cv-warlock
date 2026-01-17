"""OpenAI LLM provider."""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from cv_warlock.llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.model = model
        self.api_key = api_key

    def get_chat_model(self, temperature: float = 0.3) -> BaseChatModel:
        """Get OpenAI chat model."""
        return ChatOpenAI(
            model=self.model,
            temperature=temperature,
            api_key=self.api_key,
        )

    def get_extraction_model(self) -> BaseChatModel:
        """Get model for structured extraction with temperature=0."""
        return ChatOpenAI(
            model=self.model,
            temperature=0,
            api_key=self.api_key,
        )
