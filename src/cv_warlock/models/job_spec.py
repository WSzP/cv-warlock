"""Job specification data models."""

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, ValidationInfo, field_validator

SeniorityLevel = Literal["entry", "mid", "senior", "lead", "executive"]
JobType = Literal["full-time", "part-time", "contract", "freelance"]
RemotePolicy = Literal["remote", "hybrid", "onsite"]

_CATEGORICAL_VALUES: dict[str, frozenset[str]] = {
    "seniority_level": frozenset(get_args(SeniorityLevel)),
    "job_type": frozenset(get_args(JobType)),
    "remote": frozenset(get_args(RemotePolicy)),
}

# Spelling variants of an allowed value, after lowercasing and joining words
# with hyphens. Only true synonyms belong here; posting vocabulary with a
# different meaning (e.g. EU tender "far-site") must stay unmapped.
_CATEGORICAL_ALIASES: dict[str, str] = {
    "fulltime": "full-time",
    "parttime": "part-time",
    "on-site": "onsite",
}


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
    seniority_level: SeniorityLevel | None = None
    job_type: JobType | None = None
    remote: RemotePolicy | None = None

    @field_validator("seniority_level", "job_type", "remote", mode="before")
    @classmethod
    def coerce_categorical(cls, v: Any, info: ValidationInfo) -> str | None:
        """Normalise spelling variants, and drop values outside the allowed set.

        LLMs copy posting vocabulary verbatim (e.g. 'Far-site', 'Advanced').
        Rejecting it would discard every other field extracted from the job.
        """
        if not isinstance(v, str):
            return None
        normalised = "-".join(v.strip().lower().replace("_", " ").split())
        normalised = _CATEGORICAL_ALIASES.get(normalised, normalised)
        if normalised in _CATEGORICAL_VALUES[str(info.field_name)]:
            return normalised
        return None

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
