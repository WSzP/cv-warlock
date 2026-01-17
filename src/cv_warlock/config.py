"""Configuration management for CV Warlock."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="CV_WARLOCK_",
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys (no prefix, standard env vars)
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")

    # Provider configuration
    provider: Literal["openai", "anthropic", "google"] = "anthropic"
    model: str = "gpt-4o"
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)


    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # LangSmith tracing (no prefix - standard env vars)
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT"
    )
    langsmith_project: str = Field(default="cv-warlock", alias="LANGSMITH_PROJECT")

    @property
    def langsmith_enabled(self) -> bool:
        """Check if LangSmith tracing is enabled and configured."""
        return bool(self.langsmith_api_key and self.langsmith_tracing)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
