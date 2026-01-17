"""CV data models."""

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _coerce_to_list(v: Any) -> list[str]:
    """Coerce various inputs to list of strings for LLM output robustness."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(item) for item in v]
    if isinstance(v, str):
        # Try to parse as JSON array first
        v = v.strip()
        if v.startswith("["):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
        # Treat as comma-separated or single item
        if "," in v:
            return [item.strip() for item in v.split(",") if item.strip()]
        return [v] if v else []
    return [str(v)]


def _coerce_to_model_list(v: Any) -> list[Any]:
    """Coerce JSON string to list of dicts for nested model parsing."""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        v = v.strip()
        if v.startswith("["):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return []
    return []


class ContactInfo(BaseModel):
    """Contact information extracted from CV."""

    name: str
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None
    website: str | None = None


class Experience(BaseModel):
    """Work experience entry."""

    title: str
    company: str
    start_date: str
    end_date: str | None = None
    description: str = ""
    achievements: list[str] = Field(default_factory=list)
    skills_used: list[str] = Field(default_factory=list)

    @field_validator("achievements", "skills_used", mode="before")
    @classmethod
    def coerce_to_list(cls, v: Any) -> list[str]:
        return _coerce_to_list(v)


class Education(BaseModel):
    """Education entry."""

    degree: str
    institution: str
    graduation_date: str
    gpa: str | None = None
    relevant_coursework: list[str] = Field(default_factory=list)

    @field_validator("relevant_coursework", mode="before")
    @classmethod
    def coerce_to_list(cls, v: Any) -> list[str]:
        return _coerce_to_list(v)


class Project(BaseModel):
    """Project entry."""

    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    url: str | None = None

    @field_validator("technologies", mode="before")
    @classmethod
    def coerce_to_list(cls, v: Any) -> list[str]:
        return _coerce_to_list(v)


class Certification(BaseModel):
    """Certification or credential."""

    name: str
    issuer: str
    date: str | None = None
    url: str | None = None


class CVData(BaseModel):
    """Structured representation of a CV."""

    contact: ContactInfo
    summary: str | None = None
    experiences: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)

    @field_validator("skills", "languages", mode="before")
    @classmethod
    def coerce_to_list(cls, v: Any) -> list[str]:
        return _coerce_to_list(v)

    @field_validator("experiences", "education", "projects", "certifications", mode="before")
    @classmethod
    def coerce_model_lists(cls, v: Any) -> list[Any]:
        return _coerce_to_model_list(v)
