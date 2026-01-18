"""Tests for JobRequirements data model."""

import pytest

from cv_warlock.models.job_spec import JobRequirements


class TestJobRequirements:
    """Tests for JobRequirements model."""

    def test_minimal_job_requirements(self) -> None:
        job = JobRequirements(job_title="Software Engineer")
        assert job.job_title == "Software Engineer"
        assert job.company is None
        assert job.required_skills == []
        assert job.preferred_skills == []

    def test_full_job_requirements(self) -> None:
        job = JobRequirements(
            job_title="Senior Software Engineer",
            company="Tech Corp",
            required_skills=["Python", "AWS", "Docker"],
            preferred_skills=["Kubernetes", "Terraform"],
            required_experience_years=5,
            required_education="Bachelor's degree",
            seniority_level="senior",
            job_type="full-time",
            remote="hybrid",
            keywords=["microservices", "scalability"],
            industry_terms=["SaaS", "B2B"],
            soft_skills=["leadership", "communication"],
            company_values=["innovation", "collaboration"],
            responsibilities=["Design systems", "Lead team"],
            benefits=["Health insurance", "401k"],
        )
        assert job.company == "Tech Corp"
        assert len(job.required_skills) == 3
        assert job.required_experience_years == 5
        assert job.seniority_level == "senior"
        assert job.job_type == "full-time"
        assert job.remote == "hybrid"

    def test_seniority_level_literals(self) -> None:
        """Test that seniority_level accepts valid literals."""
        for level in ["entry", "mid", "senior", "lead", "executive"]:
            job = JobRequirements(job_title="Engineer", seniority_level=level)
            assert job.seniority_level == level

    def test_seniority_level_invalid_raises(self) -> None:
        """Test that invalid seniority_level raises ValidationError."""
        with pytest.raises(ValueError):
            JobRequirements(job_title="Engineer", seniority_level="invalid")

    def test_job_type_literals(self) -> None:
        """Test that job_type accepts valid literals."""
        for job_type in ["full-time", "part-time", "contract", "freelance"]:
            job = JobRequirements(job_title="Engineer", job_type=job_type)
            assert job.job_type == job_type

    def test_job_type_invalid_raises(self) -> None:
        """Test that invalid job_type raises ValidationError."""
        with pytest.raises(ValueError):
            JobRequirements(job_title="Engineer", job_type="invalid")

    def test_remote_literals(self) -> None:
        """Test that remote accepts valid literals."""
        for remote in ["remote", "hybrid", "onsite"]:
            job = JobRequirements(job_title="Engineer", remote=remote)
            assert job.remote == remote

    def test_remote_invalid_raises(self) -> None:
        """Test that invalid remote raises ValidationError."""
        with pytest.raises(ValueError):
            JobRequirements(job_title="Engineer", remote="invalid")

    def test_lists_default_to_empty(self) -> None:
        """Test that all list fields default to empty lists."""
        job = JobRequirements(job_title="Engineer")
        assert job.required_skills == []
        assert job.preferred_skills == []
        assert job.keywords == []
        assert job.industry_terms == []
        assert job.soft_skills == []
        assert job.company_values == []
        assert job.responsibilities == []
        assert job.benefits == []

    def test_optional_fields_default_to_none(self) -> None:
        """Test that optional fields default to None."""
        job = JobRequirements(job_title="Engineer")
        assert job.company is None
        assert job.required_experience_years is None
        assert job.required_education is None
        assert job.seniority_level is None
        assert job.job_type is None
        assert job.remote is None
