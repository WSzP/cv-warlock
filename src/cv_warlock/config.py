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

    # Provider configuration
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4o"
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
