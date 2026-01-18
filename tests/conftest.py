"""Pytest configuration and fixtures."""

from pathlib import Path
from typing import Any

import pytest

from cv_warlock.models.cv import CVData, ContactInfo, Education, Experience
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.models.state import CVWarlockState, MatchAnalysis, TailoringPlan


@pytest.fixture
def sample_cv_text() -> str:
    """Load sample CV text."""
    sample_path = Path(__file__).parent.parent / "examples" / "sample_cv.md"
    return sample_path.read_text(encoding="utf-8")


@pytest.fixture
def sample_job_text() -> str:
    """Load sample job posting text."""
    sample_path = Path(__file__).parent.parent / "examples" / "sample_job_posting.md"
    return sample_path.read_text(encoding="utf-8")


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_contact() -> ContactInfo:
    """Create a sample ContactInfo."""
    return ContactInfo(
        name="John Doe",
        email="john.doe@example.com",
        phone="+1234567890",
        location="San Francisco, CA",
        linkedin="linkedin.com/in/johndoe",
        github="github.com/johndoe",
    )


@pytest.fixture
def sample_experience() -> Experience:
    """Create a sample Experience."""
    return Experience(
        title="Senior Software Engineer",
        company="Tech Corp",
        start_date="January 2020",
        end_date="Present",
        description="Led development of cloud-native applications.",
        achievements=[
            "Led team of 5 engineers",
            "Reduced deployment time by 50%",
            "Implemented CI/CD pipeline",
        ],
        skills_used=["Python", "AWS", "Docker", "Kubernetes"],
    )


@pytest.fixture
def sample_education() -> Education:
    """Create a sample Education."""
    return Education(
        degree="Master of Science in Computer Science",
        institution="Stanford University",
        graduation_date="May 2019",
        gpa="3.9",
        relevant_coursework=["Machine Learning", "Distributed Systems", "Algorithms"],
    )


@pytest.fixture
def sample_cv_data(
    sample_contact: ContactInfo,
    sample_experience: Experience,
    sample_education: Education,
) -> CVData:
    """Create a sample CVData."""
    return CVData(
        contact=sample_contact,
        summary="Experienced software engineer with 10+ years in building scalable systems.",
        experiences=[sample_experience],
        education=[sample_education],
        skills=["Python", "AWS", "Docker", "Kubernetes", "PostgreSQL", "Redis"],
        languages=["English", "Spanish"],
    )


@pytest.fixture
def sample_job_requirements() -> JobRequirements:
    """Create sample JobRequirements."""
    return JobRequirements(
        job_title="Senior Software Engineer",
        company="Acme Inc",
        required_skills=["Python", "AWS", "Docker"],
        preferred_skills=["Kubernetes", "Terraform", "Go"],
        required_experience_years=5,
        required_education="Bachelor's degree in Computer Science",
        seniority_level="senior",
        job_type="full-time",
        remote="hybrid",
        keywords=["microservices", "cloud-native", "scalability"],
        industry_terms=["SaaS", "B2B"],
        soft_skills=["leadership", "communication", "problem-solving"],
        responsibilities=[
            "Design and implement scalable systems",
            "Lead technical projects",
            "Mentor junior engineers",
        ],
    )


@pytest.fixture
def sample_match_analysis() -> MatchAnalysis:
    """Create a sample MatchAnalysis."""
    return {
        "strong_matches": ["Python", "AWS", "Docker", "Leadership"],
        "partial_matches": ["Kubernetes", "Cloud experience"],
        "gaps": ["Terraform", "Go"],
        "transferable_skills": ["Redis experience applicable to caching"],
        "relevance_score": 0.78,
    }


@pytest.fixture
def sample_tailoring_plan() -> TailoringPlan:
    """Create a sample TailoringPlan."""
    return {
        "summary_focus": ["Cloud-native development", "Team leadership"],
        "experiences_to_emphasize": ["Senior Software Engineer at Tech Corp"],
        "skills_to_highlight": ["Python", "AWS", "Docker", "Kubernetes"],
        "achievements_to_feature": ["Led team of 5", "Reduced deployment time by 50%"],
        "keywords_to_incorporate": ["microservices", "scalability", "cloud-native"],
        "sections_to_reorder": ["summary", "experience", "skills", "education"],
    }


@pytest.fixture
def sample_state(
    sample_cv_data: CVData,
    sample_job_requirements: JobRequirements,
    sample_match_analysis: MatchAnalysis,
    sample_tailoring_plan: TailoringPlan,
) -> CVWarlockState:
    """Create a sample CVWarlockState with populated data."""
    return {
        "raw_cv": "# John Doe\n\nSenior Software Engineer...",
        "raw_job_spec": "# Senior Software Engineer\n\nWe are looking for...",
        "assume_all_tech_skills": True,
        "use_cot": True,
        "lookback_years": 4,
        "cv_data": sample_cv_data,
        "job_requirements": sample_job_requirements,
        "match_analysis": sample_match_analysis,
        "tailoring_plan": sample_tailoring_plan,
        "tailored_summary": "Experienced senior software engineer...",
        "tailored_experiences": ["**Senior Software Engineer at Tech Corp**..."],
        "tailored_skills": ["Python", "AWS", "Docker", "Kubernetes"],
        "tailored_cv": "# John Doe\n\n## Summary\n\nExperienced senior...",
        "summary_reasoning_result": None,
        "experience_reasoning_results": None,
        "skills_reasoning_result": None,
        "generation_context": None,
        "total_refinement_iterations": 1,
        "quality_scores": {"summary": "EXCELLENT", "experiences": "GOOD"},
        "step_timings": [],
        "current_step_start": None,
        "total_generation_time": 45.5,
        "messages": [],
        "current_step": "complete",
        "current_step_description": "CV tailoring complete",
        "errors": [],
    }


@pytest.fixture
def empty_state() -> CVWarlockState:
    """Create an empty/initial CVWarlockState."""
    return {
        "raw_cv": "",
        "raw_job_spec": "",
        "assume_all_tech_skills": True,
        "use_cot": True,
        "lookback_years": None,
        "cv_data": None,
        "job_requirements": None,
        "match_analysis": None,
        "tailoring_plan": None,
        "tailored_summary": None,
        "tailored_experiences": None,
        "tailored_skills": None,
        "tailored_cv": None,
        "summary_reasoning_result": None,
        "experience_reasoning_results": None,
        "skills_reasoning_result": None,
        "generation_context": None,
        "total_refinement_iterations": 0,
        "quality_scores": None,
        "step_timings": [],
        "current_step_start": None,
        "total_generation_time": None,
        "messages": [],
        "current_step": "initialize",
        "current_step_description": "Starting workflow",
        "errors": [],
    }
