"""Tests for JobRequirements data model."""

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

    def test_seniority_level_invalid_becomes_none(self) -> None:
        """An unrecognised seniority_level is dropped, not fatal to the extraction."""
        job = JobRequirements(job_title="Engineer", seniority_level="invalid")  # type: ignore[arg-type]
        assert job.seniority_level is None

    def test_job_type_literals(self) -> None:
        """Test that job_type accepts valid literals."""
        for job_type in ["full-time", "part-time", "contract", "freelance"]:
            job = JobRequirements(job_title="Engineer", job_type=job_type)
            assert job.job_type == job_type

    def test_job_type_invalid_becomes_none(self) -> None:
        """An unrecognised job_type is dropped, not fatal to the extraction."""
        job = JobRequirements(job_title="Engineer", job_type="invalid")  # type: ignore[arg-type]
        assert job.job_type is None

    def test_remote_literals(self) -> None:
        """Test that remote accepts valid literals."""
        for remote in ["remote", "hybrid", "onsite"]:
            job = JobRequirements(job_title="Engineer", remote=remote)
            assert job.remote == remote

    def test_remote_invalid_becomes_none(self) -> None:
        """An unrecognised remote value is dropped, not fatal to the extraction."""
        job = JobRequirements(job_title="Engineer", remote="invalid")  # type: ignore[arg-type]
        assert job.remote is None

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


class TestCategoricalFieldValidators:
    """Tests for seniority_level, job_type and remote coercion.

    These fields are closed sets, but job postings use vocabulary the sets do
    not cover. A value the model copies verbatim from the posting must not
    discard every other field that was extracted correctly.
    """

    def test_far_site_posting_extracts_instead_of_failing(self) -> None:
        """Regression: an EU tender marked 'Far-site' failed the whole job extraction.

        The payload mirrors what gpt-5-mini returned for the DGT T.2 posting.
        'Far-site' means the contractor's premises, which is not the same as
        remote work, so it is dropped rather than guessed.
        """
        job = JobRequirements.model_validate(
            {
                "job_title": "Back-end Developer (Advanced)",
                "company": "DGT T.2",
                "required_skills": ["C++", "TypeScript", "Python", "Bash", "Perl"],
                "required_experience_years": 10,
                "seniority_level": "Advanced",
                "job_type": "Far-site",
                "remote": "Far-site",
            }
        )
        assert job.job_title == "Back-end Developer (Advanced)"
        assert job.required_skills == ["C++", "TypeScript", "Python", "Bash", "Perl"]
        assert job.required_experience_years == 10
        assert job.remote is None
        assert job.job_type is None
        assert job.seniority_level is None

    def test_case_and_whitespace_are_normalised(self) -> None:
        job = JobRequirements(
            job_title="Engineer",
            seniority_level=" Senior ",  # type: ignore[arg-type]
            job_type="Full-Time",  # type: ignore[arg-type]
            remote="HYBRID",  # type: ignore[arg-type]
        )
        assert job.seniority_level == "senior"
        assert job.job_type == "full-time"
        assert job.remote == "hybrid"

    def test_spelling_variants_map_to_the_canonical_value(self) -> None:
        for variant in ["on-site", "On site", "on_site"]:
            job = JobRequirements(job_title="Engineer", remote=variant)  # type: ignore[arg-type]
            assert job.remote == "onsite", variant
        for variant in ["full time", "Full_Time", "fulltime"]:
            job = JobRequirements(job_title="Engineer", job_type=variant)  # type: ignore[arg-type]
            assert job.job_type == "full-time", variant
        for variant in ["part time", "parttime"]:
            job = JobRequirements(job_title="Engineer", job_type=variant)  # type: ignore[arg-type]
            assert job.job_type == "part-time", variant

    def test_non_string_values_become_none(self) -> None:
        job = JobRequirements(
            job_title="Engineer",
            seniority_level=3,  # type: ignore[arg-type]
            job_type=["contract"],  # type: ignore[arg-type]
            remote=True,  # type: ignore[arg-type]
        )
        assert job.seniority_level is None
        assert job.job_type is None
        assert job.remote is None


class TestExperienceYearsValidator:
    """Tests for coerce_experience_years validator."""

    def test_none_returns_none(self) -> None:
        """Test that None input returns None."""
        job = JobRequirements(job_title="Engineer", required_experience_years=None)
        assert job.required_experience_years is None

    def test_int_returns_int(self) -> None:
        """Test that integer input is preserved."""
        job = JobRequirements(job_title="Engineer", required_experience_years=5)
        assert job.required_experience_years == 5

    def test_valid_string_parses_to_int(self) -> None:
        """Test that valid numeric string is parsed to int."""
        job = JobRequirements(job_title="Engineer", required_experience_years="3")
        assert job.required_experience_years == 3

    def test_unknown_string_returns_none(self) -> None:
        """Test that '<UNKNOWN>' string returns None."""
        job = JobRequirements(job_title="Engineer", required_experience_years="<UNKNOWN>")
        assert job.required_experience_years is None

    def test_not_specified_string_returns_none(self) -> None:
        """Test that 'Not specified' string returns None."""
        job = JobRequirements(job_title="Engineer", required_experience_years="Not specified")
        assert job.required_experience_years is None

    def test_empty_string_returns_none(self) -> None:
        """Test that empty string returns None."""
        job = JobRequirements(job_title="Engineer", required_experience_years="")
        assert job.required_experience_years is None

    def test_whitespace_string_returns_none(self) -> None:
        """Test that whitespace-only string returns None."""
        job = JobRequirements(job_title="Engineer", required_experience_years="   ")
        assert job.required_experience_years is None

    def test_float_coerced_to_none(self) -> None:
        """Test that float input is coerced to None (not an int or str)."""
        job = JobRequirements(job_title="Engineer", required_experience_years=5.5)  # type: ignore[arg-type]
        assert job.required_experience_years is None


class TestListFieldValidators:
    """Tests for ensure_list validator on list fields."""

    def test_string_value_becomes_empty_list(self) -> None:
        """Test that string values like 'Not specified' become empty lists."""
        job = JobRequirements(
            job_title="Engineer",
            required_skills="Not specified",  # type: ignore[arg-type]
            keywords="N/A",  # type: ignore[arg-type]
        )
        assert job.required_skills == []
        assert job.keywords == []

    def test_none_becomes_empty_list(self) -> None:
        """Test that None values become empty lists."""
        job = JobRequirements(
            job_title="Engineer",
            required_skills=None,  # type: ignore[arg-type]
        )
        assert job.required_skills == []

    def test_list_preserved(self) -> None:
        """Test that valid list values are preserved."""
        skills = ["Python", "AWS"]
        job = JobRequirements(job_title="Engineer", required_skills=skills)
        assert job.required_skills == skills
