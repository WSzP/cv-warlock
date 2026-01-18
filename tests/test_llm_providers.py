"""Tests for LLM provider abstraction."""

from unittest.mock import MagicMock, patch

import pytest

from cv_warlock.llm.base import LLMProvider, get_llm_provider


class TestGetLLMProvider:
    """Tests for the get_llm_provider factory function."""

    @patch("cv_warlock.llm.openai.OpenAIProvider")
    def test_openai_provider(self, mock_provider: MagicMock) -> None:
        """Test that openai provider is correctly instantiated."""
        mock_instance = MagicMock()
        mock_provider.return_value = mock_instance

        result = get_llm_provider("openai", model="gpt-5.2", api_key="test-key")

        mock_provider.assert_called_once_with(model="gpt-5.2", api_key="test-key")
        assert result is mock_instance

    @patch("cv_warlock.llm.anthropic.AnthropicProvider")
    def test_anthropic_provider(self, mock_provider: MagicMock) -> None:
        """Test that anthropic provider is correctly instantiated."""
        mock_instance = MagicMock()
        mock_provider.return_value = mock_instance

        result = get_llm_provider(
            "anthropic", model="claude-sonnet-4-5-20250929", api_key="test-key"
        )

        mock_provider.assert_called_once_with(
            model="claude-sonnet-4-5-20250929", api_key="test-key"
        )
        assert result is mock_instance

    @patch("cv_warlock.llm.google.GoogleProvider")
    def test_google_provider(self, mock_provider: MagicMock) -> None:
        """Test that google provider is correctly instantiated."""
        mock_instance = MagicMock()
        mock_provider.return_value = mock_instance

        result = get_llm_provider("google", model="gemini-3-flash-preview", api_key="test-key")

        mock_provider.assert_called_once_with(model="gemini-3-flash-preview", api_key="test-key")
        assert result is mock_instance

    @patch("cv_warlock.llm.openai.OpenAIProvider")
    def test_openai_default_model(self, mock_provider: MagicMock) -> None:
        """Test that openai uses default model when not specified."""
        get_llm_provider("openai")
        mock_provider.assert_called_once_with(model="gpt-5.2", api_key=None)

    @patch("cv_warlock.llm.anthropic.AnthropicProvider")
    def test_anthropic_default_model(self, mock_provider: MagicMock) -> None:
        """Test that anthropic uses default model when not specified."""
        get_llm_provider("anthropic")
        mock_provider.assert_called_once_with(model="claude-sonnet-4-5-20250929", api_key=None)

    @patch("cv_warlock.llm.google.GoogleProvider")
    def test_google_default_model(self, mock_provider: MagicMock) -> None:
        """Test that google uses default model when not specified."""
        get_llm_provider("google")
        mock_provider.assert_called_once_with(model="gemini-3-flash-preview", api_key=None)

    def test_unknown_provider_raises(self) -> None:
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider: invalid"):
            get_llm_provider("invalid")  # type: ignore[arg-type]


class TestLLMProviderInterface:
    """Tests for the LLMProvider abstract base class."""

    def test_is_abstract(self) -> None:
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_abstract_methods(self) -> None:
        """Test that abstract methods are defined."""
        assert hasattr(LLMProvider, "get_chat_model")
        assert hasattr(LLMProvider, "get_extraction_model")
        assert hasattr(LLMProvider, "extract_structured")


class TestLLMProviderExtractStructured:
    """Tests for the extract_structured method."""

    def test_extract_structured_calls_chain(self) -> None:
        """Test that extract_structured creates the correct chain."""

        # Create a concrete implementation for testing
        class MockProvider(LLMProvider):
            def __init__(self) -> None:
                self.mock_model = MagicMock()

            def get_chat_model(self):
                return self.mock_model

            def get_extraction_model(self):
                return self.mock_model

        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            value: int

        provider = MockProvider()
        structured_mock = MagicMock()
        structured_mock.invoke.return_value = TestSchema(name="test", value=42)
        provider.mock_model.with_structured_output.return_value = structured_mock

        result = provider.extract_structured("test prompt", TestSchema)

        provider.mock_model.with_structured_output.assert_called_once_with(
            TestSchema, method="function_calling"
        )
        structured_mock.invoke.assert_called_once_with("test prompt")
        assert result.name == "test"
        assert result.value == 42
