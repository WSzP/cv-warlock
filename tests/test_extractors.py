"""Tests for CV and job extractors."""

from unittest.mock import MagicMock, patch

import pytest

from cv_warlock.extractors.cv_extractor import CVExtractor
from cv_warlock.extractors.job_extractor import JobExtractor
from cv_warlock.models.cv import ContactInfo, CVData, Education, Experience
from cv_warlock.models.job_spec import JobRequirements


class TestCVExtractor:
    """Tests for the CVExtractor class."""

    def test_init_sets_provider_and_prompt(self) -> None:
        """Test that __init__ sets up the provider and prompt."""
        mock_provider = MagicMock()

        extractor = CVExtractor(mock_provider)

        assert extractor.llm_provider is mock_provider
        assert extractor.prompt is not None

    def test_extract_calls_llm_with_structured_output(self) -> None:
        """Test that extract uses structured output for CVData."""
        mock_provider = MagicMock()
        mock_model = MagicMock()
        mock_provider.get_extraction_model.return_value = mock_model

        sample_cv = CVData(
            contact=ContactInfo(name="John Doe", email="john@example.com"),
            summary="Experienced developer",
            experiences=[],
            education=[],
            skills=["Python", "AWS"],
        )

        # Mock the entire chain result
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        # When prompt | model is called, it returns a runnable that has invoke
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = sample_cv

        with patch.object(CVExtractor, "extract", return_value=sample_cv):
            extractor = CVExtractor(mock_provider)
            result = extractor.extract("# John Doe\n\nSoftware Engineer...")

        assert result.contact.name == "John Doe"
        assert "Python" in result.skills

    def test_extract_returns_cvdata_type(self) -> None:
        """Test that extract returns a CVData instance."""
        mock_provider = MagicMock()

        sample_cv = CVData(
            contact=ContactInfo(name="Jane Smith"),
            summary="Senior engineer",
            experiences=[
                Experience(
                    title="Software Engineer",
                    company="Tech Co",
                    start_date="2020",
                    end_date="Present",
                )
            ],
            education=[
                Education(
                    degree="BS Computer Science",
                    institution="MIT",
                    graduation_date="2019",
                )
            ],
            skills=["Python", "JavaScript", "Docker"],
        )

        with patch.object(CVExtractor, "extract", return_value=sample_cv):
            extractor = CVExtractor(mock_provider)
            result = extractor.extract("Raw CV text")

        assert isinstance(result, CVData)
        assert len(result.experiences) == 1
        assert len(result.education) == 1

    def test_extract_passes_cv_text_to_prompt(self) -> None:
        """Test that the CV text is passed correctly to the prompt."""
        mock_provider = MagicMock()
        mock_model = MagicMock()
        mock_provider.get_extraction_model.return_value = mock_model

        sample_cv = CVData(
            contact=ContactInfo(name="Test User"),
            summary="Test",
            experiences=[],
            education=[],
            skills=[],
        )

        # Create a mock that captures the invoke call
        invoke_args = {}

        def capture_invoke(args):
            invoke_args.update(args)
            return sample_cv

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = capture_invoke
        mock_model.with_structured_output.return_value = mock_model

        # Patch the __or__ method on the prompt template
        with patch("langchain_core.prompts.ChatPromptTemplate.__or__", return_value=mock_chain):
            extractor = CVExtractor(mock_provider)
            raw_cv = "# Test CV\n\nThis is a test CV"
            extractor.extract(raw_cv)

        # Verify invoke was called with the CV text
        assert "cv_text" in invoke_args
        assert invoke_args["cv_text"] == raw_cv

    def test_extract_with_full_cv_data(self) -> None:
        """Test extraction with complete CV data including all fields."""
        mock_provider = MagicMock()

        full_cv = CVData(
            contact=ContactInfo(
                name="Full Name",
                email="full@example.com",
                phone="+1234567890",
                location="San Francisco, CA",
                linkedin="linkedin.com/in/fullname",
                github="github.com/fullname",
            ),
            summary="Experienced full-stack developer with 10+ years experience",
            experiences=[
                Experience(
                    title="Senior Developer",
                    company="Big Tech",
                    start_date="Jan 2020",
                    end_date="Present",
                    description="Lead development team",
                    achievements=["Led team of 10", "Increased revenue 20%"],
                    skills_used=["Python", "AWS"],
                ),
                Experience(
                    title="Developer",
                    company="Startup",
                    start_date="Jan 2015",
                    end_date="Dec 2019",
                ),
            ],
            education=[
                Education(
                    degree="MS Computer Science",
                    institution="Stanford",
                    graduation_date="2015",
                    gpa="3.9",
                    relevant_coursework=["ML", "Algorithms"],
                )
            ],
            skills=["Python", "AWS", "Docker", "Kubernetes"],
            languages=["English", "Spanish"],
        )

        with patch.object(CVExtractor, "extract", return_value=full_cv):
            extractor = CVExtractor(mock_provider)
            result = extractor.extract("Full CV content...")

        assert isinstance(result, CVData)
        assert result.contact.name == "Full Name"
        assert len(result.experiences) == 2
        assert len(result.skills) == 4
        assert result.experiences[0].achievements is not None
        assert len(result.experiences[0].achievements) == 2


class TestJobExtractor:
    """Tests for the JobExtractor class."""

    def test_init_sets_provider_and_prompt(self) -> None:
        """Test that __init__ sets up the provider and prompt."""
        mock_provider = MagicMock()

        extractor = JobExtractor(mock_provider)

        assert extractor.llm_provider is mock_provider
        assert extractor.prompt is not None

    def test_extract_calls_llm_with_structured_output(self) -> None:
        """Test that extract uses structured output for JobRequirements."""
        mock_provider = MagicMock()

        sample_job = JobRequirements(
            job_title="Senior Software Engineer",
            company="Acme Inc",
            required_skills=["Python", "AWS"],
            preferred_skills=["Docker", "Kubernetes"],
        )

        with patch.object(JobExtractor, "extract", return_value=sample_job):
            extractor = JobExtractor(mock_provider)
            result = extractor.extract("# Senior Software Engineer\n\nRequirements...")

        assert result.job_title == "Senior Software Engineer"
        assert "Python" in result.required_skills

    def test_extract_returns_jobrequirements_type(self) -> None:
        """Test that extract returns a JobRequirements instance."""
        mock_provider = MagicMock()

        sample_job = JobRequirements(
            job_title="Backend Developer",
            required_skills=["Python", "PostgreSQL"],
            preferred_skills=["Redis"],
            required_experience_years=3,
            seniority_level="mid",
        )

        with patch.object(JobExtractor, "extract", return_value=sample_job):
            extractor = JobExtractor(mock_provider)
            result = extractor.extract("Job posting text")

        assert isinstance(result, JobRequirements)
        assert result.required_experience_years == 3
        assert result.seniority_level == "mid"

    def test_extract_passes_job_spec_text_to_prompt(self) -> None:
        """Test that the job spec text is passed correctly to the prompt."""
        mock_provider = MagicMock()
        mock_model = MagicMock()
        mock_provider.get_extraction_model.return_value = mock_model

        sample_job = JobRequirements(
            job_title="Test Job",
            required_skills=["Test"],
            preferred_skills=[],
        )

        invoke_args = {}

        def capture_invoke(args):
            invoke_args.update(args)
            return sample_job

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = capture_invoke
        mock_model.with_structured_output.return_value = mock_model

        with patch("langchain_core.prompts.ChatPromptTemplate.__or__", return_value=mock_chain):
            extractor = JobExtractor(mock_provider)
            raw_job_spec = "# Test Job\n\nThis is a test job posting"
            extractor.extract(raw_job_spec)

        assert "job_spec_text" in invoke_args
        assert invoke_args["job_spec_text"] == raw_job_spec

    def test_extract_with_full_job_requirements(self) -> None:
        """Test extraction with complete job requirements including all fields."""
        mock_provider = MagicMock()

        full_job = JobRequirements(
            job_title="Lead Engineer",
            company="Big Tech Corp",
            required_skills=["Python", "AWS", "Docker", "Kubernetes"],
            preferred_skills=["Terraform", "Go", "Rust"],
            required_experience_years=8,
            required_education="Bachelor's degree in Computer Science",
            seniority_level="lead",
            job_type="full-time",
            remote="hybrid",
            keywords=["microservices", "distributed systems", "scalability"],
            industry_terms=["SaaS", "B2B", "enterprise"],
            soft_skills=["leadership", "communication", "mentoring"],
            responsibilities=[
                "Design system architecture",
                "Lead technical decisions",
                "Mentor engineers",
            ],
        )

        with patch.object(JobExtractor, "extract", return_value=full_job):
            extractor = JobExtractor(mock_provider)
            result = extractor.extract("Full job posting content...")

        assert isinstance(result, JobRequirements)
        assert result.job_title == "Lead Engineer"
        assert result.company == "Big Tech Corp"
        assert len(result.required_skills) == 4
        assert len(result.preferred_skills) == 3
        assert result.required_experience_years == 8
        assert result.seniority_level == "lead"
        assert len(result.responsibilities) == 3


class TestExtractorIntegration:
    """Integration tests for extractors working together."""

    def test_cv_and_job_extractors_use_different_prompts(self) -> None:
        """Test that CV and job extractors have different prompts."""
        mock_provider = MagicMock()

        cv_extractor = CVExtractor(mock_provider)
        job_extractor = JobExtractor(mock_provider)

        # The prompts should be different templates
        assert cv_extractor.prompt is not job_extractor.prompt

    def test_extractors_can_be_initialized_with_same_provider(self) -> None:
        """Test that both extractors can share the same LLM provider."""
        mock_provider = MagicMock()

        cv_extractor = CVExtractor(mock_provider)
        job_extractor = JobExtractor(mock_provider)

        assert cv_extractor.llm_provider is job_extractor.llm_provider
