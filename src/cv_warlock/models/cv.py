"""CV data models."""

from pydantic import BaseModel, Field


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


class Education(BaseModel):
    """Education entry."""

    degree: str
    institution: str
    graduation_date: str
    gpa: str | None = None
    relevant_coursework: list[str] = Field(default_factory=list)


class Project(BaseModel):
    """Project entry."""

    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    url: str | None = None


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
