"""Tests for configuration and settings."""

import os
from unittest.mock import patch

import pytest

from cv_warlock.config import Settings, get_settings


class TestSettings:
    """Tests for Settings configuration class."""

    def test_default_values(self) -> None:
        """Test that defaults are set correctly when no env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            # Note: Settings reads from .env.local file, so we mock the file loading
            settings = Settings(_env_file=None)  # Disable env file loading
            assert settings.provider == "anthropic"
            # Model is auto-selected via Dual-Model Strategy, not configured
            assert settings.lookback_years == 4
            assert settings.log_level == "INFO"
            assert settings.openai_api_key is None
            assert settings.anthropic_api_key is None
            assert settings.google_api_key is None

    def test_provider_from_env(self) -> None:
        """Test provider setting from environment variable."""
        with patch.dict(os.environ, {"CV_WARLOCK_PROVIDER": "openai"}, clear=True):
            settings = Settings()
            assert settings.provider == "openai"

    def test_lookback_years_from_env(self) -> None:
        """Test lookback_years setting from environment variable."""
        with patch.dict(os.environ, {"CV_WARLOCK_LOOKBACK_YEARS": "10"}, clear=True):
            settings = Settings()
            assert settings.lookback_years == 10

    def test_lookback_years_minimum(self) -> None:
        """Test that lookback_years has minimum of 0."""
        with patch.dict(os.environ, {"CV_WARLOCK_LOOKBACK_YEARS": "-1"}, clear=True):
            with pytest.raises(ValueError):
                Settings()

    def test_lookback_years_maximum(self) -> None:
        """Test that lookback_years has maximum of 50."""
        with patch.dict(os.environ, {"CV_WARLOCK_LOOKBACK_YEARS": "51"}, clear=True):
            with pytest.raises(ValueError):
                Settings()

    def test_lookback_years_at_boundaries(self) -> None:
        """Test lookback_years at valid boundaries."""
        with patch.dict(os.environ, {"CV_WARLOCK_LOOKBACK_YEARS": "0"}, clear=True):
            settings = Settings()
            assert settings.lookback_years == 0

        with patch.dict(os.environ, {"CV_WARLOCK_LOOKBACK_YEARS": "50"}, clear=True):
            settings = Settings()
            assert settings.lookback_years == 50

    def test_api_keys_from_env(self) -> None:
        """Test API key loading from environment variables."""
        env = {
            "OPENAI_API_KEY": "sk-test-openai",
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "GOOGLE_API_KEY": "test-google-key",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.openai_api_key == "sk-test-openai"
            assert settings.anthropic_api_key == "sk-ant-test"
            assert settings.google_api_key == "test-google-key"

    def test_log_level_values(self) -> None:
        """Test valid log level values."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            with patch.dict(os.environ, {"CV_WARLOCK_LOG_LEVEL": level}, clear=True):
                settings = Settings()
                assert settings.log_level == level

    def test_provider_invalid_raises(self) -> None:
        """Test that invalid provider raises ValidationError."""
        with patch.dict(os.environ, {"CV_WARLOCK_PROVIDER": "invalid"}, clear=True):
            with pytest.raises(ValueError):
                Settings()


class TestLangSmithSettings:
    """Tests for LangSmith configuration."""

    def test_langsmith_defaults(self) -> None:
        """Test LangSmith default values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.langsmith_api_key is None
            assert settings.langsmith_tracing is False
            assert settings.langsmith_endpoint == "https://api.smith.langchain.com"
            assert settings.langsmith_project == "cv-warlock"

    def test_langsmith_enabled_property(self) -> None:
        """Test langsmith_enabled property logic."""
        # Neither key nor tracing -> disabled
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.langsmith_enabled is False

        # Key but no tracing -> disabled
        with patch.dict(
            os.environ, {"LANGSMITH_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}, clear=True
        ):
            settings = Settings()
            assert settings.langsmith_enabled is False

        # No key but tracing -> disabled
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}, clear=True):
            settings = Settings()
            assert settings.langsmith_enabled is False

        # Both key and tracing -> enabled
        with patch.dict(
            os.environ, {"LANGSMITH_API_KEY": "test-key", "LANGSMITH_TRACING": "true"}, clear=True
        ):
            settings = Settings()
            assert settings.langsmith_enabled is True

    def test_langsmith_custom_endpoint(self) -> None:
        """Test custom LangSmith endpoint."""
        env = {"LANGSMITH_ENDPOINT": "https://eu.api.smith.langchain.com"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.langsmith_endpoint == "https://eu.api.smith.langchain.com"

    def test_langsmith_custom_project(self) -> None:
        """Test custom LangSmith project name."""
        env = {"LANGSMITH_PROJECT": "my-custom-project"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.langsmith_project == "my-custom-project"


class TestGetSettings:
    """Tests for get_settings factory function."""

    def test_returns_settings_instance(self) -> None:
        """Test that get_settings returns a Settings instance."""
        # Clear cache to ensure fresh instance
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_caching(self) -> None:
        """Test that get_settings returns cached instance."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
