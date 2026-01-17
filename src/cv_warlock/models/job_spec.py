"""Job specification data models."""

from typing import Literal

from pydantic import BaseModel, Field


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
