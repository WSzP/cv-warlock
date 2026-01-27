"""Job specification data models."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class JobRequirements(BaseModel):
    """Extracted requirements from a job specification."""

    job_title: str
    company: str | None = None

    # Required qualifications
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    required_experience_years: int | None = None
    required_education: str | None = None

    # Job characteristics
    seniority_level: Literal["entry", "mid", "senior", "lead", "executive"] | None = None
    job_type: Literal["full-time", "part-time", "contract", "freelance"] | None = None
    remote: Literal["remote", "hybrid", "onsite"] | None = None

    # Keywords and themes
    keywords: list[str] = Field(default_factory=list)
    industry_terms: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)

    # Culture and values
    company_values: list[str] = Field(default_factory=list)

    # Raw sections for context
    responsibilities: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)

    @field_validator("required_experience_years", mode="before")
    @classmethod
    def coerce_experience_years(cls, v: Any) -> int | None:
        """Convert invalid values (like '<UNKNOWN>') to None."""
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            # Try to parse as integer
            try:
                return int(v)
            except ValueError:
                # LLM returned something like '<UNKNOWN>' or 'Not specified'
                return None
        return None

    @field_validator(
        "required_skills",
        "preferred_skills",
        "keywords",
        "industry_terms",
        "soft_skills",
        "company_values",
        "responsibilities",
        "benefits",
        mode="before",
    )
    @classmethod
    def ensure_list(cls, v: Any) -> list[str]:
        """Convert non-list values (like 'Not specified') to empty list."""
        if v is None:
            return []
        if isinstance(v, str):
            # LLM sometimes returns "Not specified" or similar strings
            return []
        if isinstance(v, list):
            return v
        return []
