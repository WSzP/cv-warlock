"""Anthropic Claude LLM provider."""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from cv_warlock.llm.base import LLMProvider

# Default timeout in seconds for API requests
# Set conservative timeout to prevent hanging on slow responses
DEFAULT_TIMEOUT = 120.0  # 2 minutes per request
DEFAULT_MAX_RETRIES = 3  # 3 retries = 4 total attempts (handles transient connection errors)

# Prompt caching reduces costs by 90% for repeated content
# See: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
PROMPT_CACHING_HEADER = {"anthropic-beta": "prompt-caching-2024-07-31"}


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with model caching."""

    def __init__(
        self,
        model: str = "claude-opus-4-5-20251101",
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        # Initialize cache attributes from parent
        self._chat_model = None
        self._extraction_model = None

    def _create_chat_model(self) -> BaseChatModel:
        """Create Anthropic chat model with prompt caching enabled."""
        return ChatAnthropic(
            model=self.model,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=PROMPT_CACHING_HEADER,
        )

    def _create_extraction_model(self) -> BaseChatModel:
        """Create model for structured extraction with prompt caching enabled."""
        return ChatAnthropic(
            model=self.model,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=PROMPT_CACHING_HEADER,
        )
