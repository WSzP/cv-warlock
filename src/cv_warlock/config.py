"""Configuration management for CV Warlock."""

from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from cv_warlock.rlm.models import RLMConfig


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
    model: str = "claude-sonnet-4-5-20250929"

    # Tailoring configuration
    lookback_years: int = Field(
        default=4,
        ge=0,
        le=50,
        description="Only tailor experiences that ended within this many years",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # LangSmith tracing (no prefix - standard env vars)
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT"
    )
    langsmith_project: str = Field(default="cv-warlock", alias="LANGSMITH_PROJECT")

    # RLM (Recursive Language Model) Configuration
    rlm_enabled: bool = Field(
        default=True,
        description="Enable RLM for handling large CVs and job specs",
    )
    rlm_size_threshold: int = Field(
        default=8000,
        ge=1000,
        le=100000,
        description="Character count threshold to automatically trigger RLM mode",
    )
    rlm_root_provider: Literal["openai", "anthropic", "google"] = Field(
        default="anthropic",
        description="Provider for RLM root model (needs strong coding ability)",
    )
    rlm_root_model: str = Field(
        default="claude-opus-4-5-20251101",
        description="Model ID for RLM root model",
    )
    rlm_sub_provider: Literal["openai", "anthropic", "google"] | None = Field(
        default=None,
        description="Provider for RLM sub-calls (defaults to root provider)",
    )
    rlm_sub_model: str | None = Field(
        default=None,
        description="Model for sub-calls (defaults to faster model)",
    )
    rlm_max_iterations: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum orchestrator iterations per analysis",
    )
    rlm_max_sub_calls: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum sub-LLM calls per analysis",
    )
    rlm_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=600,
        description="Total timeout for RLM analysis in seconds",
    )
    rlm_sandbox_mode: Literal["local", "docker", "modal"] = Field(
        default="local",
        description="Sandbox mode for code execution (local for dev, docker for prod)",
    )

    @property
    def langsmith_enabled(self) -> bool:
        """Check if LangSmith tracing is enabled and configured."""
        return bool(self.langsmith_api_key and self.langsmith_tracing)

    @property
    def rlm_config(self) -> "RLMConfig":
        """Get RLM configuration object."""
        from cv_warlock.rlm.models import RLMConfig

        return RLMConfig(
            root_provider=self.rlm_root_provider,
            root_model=self.rlm_root_model,
            sub_provider=self.rlm_sub_provider,
            sub_model=self.rlm_sub_model,
            max_iterations=self.rlm_max_iterations,
            max_sub_calls=self.rlm_max_sub_calls,
            timeout_seconds=self.rlm_timeout_seconds,
            size_threshold=self.rlm_size_threshold,
            sandbox_mode=self.rlm_sandbox_mode,
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
