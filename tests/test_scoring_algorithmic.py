"""Tests for the AlgorithmicScorer."""

from datetime import datetime

import pytest

from cv_warlock.models.cv import ContactInfo, CVData, Education, Experience
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.scoring.algorithmic import AlgorithmicScorer
from cv_warlock.scoring.models import AlgorithmicScores


@pytest.fixture
def scorer() -> AlgorithmicScorer:
    """Create an AlgorithmicScorer instance."""
    return AlgorithmicScorer()


@pytest.fixture
def sample_cv_data() -> CVData:
    """Create sample CV data for testing."""
    return CVData(
        contact=ContactInfo(name="John Doe", email="john@example.com"),
        summary="Experienced software engineer with 10+ years building scalable systems.",
        experiences=[
            Experience(
                title="Senior Software Engineer",
                company="Tech Corp",
                start_date="January 2020",
                end_date="Present",
                description="Led development of cloud-native applications using Python and AWS",
                achievements=[
                    "Led team of 5 engineers",
                    "Reduced deployment time by 50%",
                    "Implemented CI/CD pipeline with Docker",
                ],
                skills_used=["Python", "AWS", "Docker"],
            ),
            Experience(
                title="Software Engineer",
                company="Startup Inc",
                start_date="January 2015",
                end_date="December 2019",
                description="Built microservices architecture",
                achievements=["Increased performance by 40%"],
                skills_used=["Python", "PostgreSQL"],
            ),
        ],
        education=[
            Education(
                degree="Master of Science in Computer Science",
                institution="Stanford University",
                graduation_date="2015",
                gpa="3.9",
            )
        ],
        skills=["Python", "AWS", "Docker", "Kubernetes", "PostgreSQL", "Redis"],
    )


@pytest.fixture
def sample_job_requirements() -> JobRequirements:
    """Create sample job requirements for testing."""
    return JobRequirements(
        job_title="Senior Software Engineer",
        company="Acme Inc",
        required_skills=["Python", "AWS", "Docker"],
        preferred_skills=["Kubernetes", "Terraform"],
        required_experience_years=5,
        required_education="Bachelor's degree in Computer Science",
        seniority_level="senior",
        responsibilities=["Design systems", "Lead projects"],
    )


class TestAlgorithmicScorerWeights:
    """Tests for AlgorithmicScorer weight configuration."""

    def test_weights_sum_to_one(self, scorer: AlgorithmicScorer) -> None:
        """Test that weight values sum to approximately 1.0."""
        total_weight = sum(scorer.WEIGHTS.values())
        assert abs(total_weight - 1.0) < 0.01  # Allow small float error

    def test_weights_contain_expected_keys(self, scorer: AlgorithmicScorer) -> None:
        """Test that WEIGHTS contains expected keys."""
        expected_keys = {
            "exact_skill_match",
            "experience_years_fit",
            "education_match",
            "recency_score",
        }
        assert set(scorer.WEIGHTS.keys()) == expected_keys

    def test_all_weights_positive(self, scorer: AlgorithmicScorer) -> None:
        """Test that all weights are positive."""
        for weight in scorer.WEIGHTS.values():
            assert weight > 0


class TestAlgorithmicScorerCompute:
    """Tests for the main compute method."""

    def test_compute_returns_algorithmic_scores(
        self,
        scorer: AlgorithmicScorer,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that compute returns AlgorithmicScores."""
        result = scorer.compute(sample_cv_data, sample_job_requirements)

        assert isinstance(result, AlgorithmicScores)

    def test_compute_all_scores_in_range(
        self,
        scorer: AlgorithmicScorer,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that all computed scores are in [0, 1] range."""
        result = scorer.compute(sample_cv_data, sample_job_requirements)

        assert 0 <= result.exact_skill_match <= 1
        assert 0 <= result.semantic_skill_match <= 1
        assert 0 <= result.document_similarity <= 1
        assert 0 <= result.experience_years_fit <= 1
        assert 0 <= result.education_match <= 1
        assert 0 <= result.recency_score <= 1
        assert 0 <= result.total <= 1

    def test_compute_with_matching_skills(
        self,
        scorer: AlgorithmicScorer,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test compute with CV that has matching required skills."""
        result = scorer.compute(sample_cv_data, sample_job_requirements)

        # All required skills (Python, AWS, Docker) are in CV
        assert result.exact_skill_match > 0.7
        assert result.knockout_triggered is False

    def test_compute_with_knockout(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test compute with CV missing required skills (knockout)."""
        cv_data = CVData(
            contact=ContactInfo(name="Jane Doe"),
            skills=["Java", "Spring"],  # No Python, AWS, or Docker
            experiences=[],
            education=[],
        )

        job_requirements = JobRequirements(
            job_title="Python Engineer",
            required_skills=["Python", "AWS", "Docker"],
            preferred_skills=[],
        )

        result = scorer.compute(cv_data, job_requirements)

        assert result.knockout_triggered is True
        assert result.knockout_reason is not None
        assert "Missing required skills" in result.knockout_reason
        assert result.total == 0.0

    def test_compute_knockout_zeros_total(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that knockout sets total to 0."""
        cv_data = CVData(
            contact=ContactInfo(name="Jane Doe"),
            skills=[],
            experiences=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date="2020",
                    end_date="Present",
                    description="Did work",
                    skills_used=["JavaScript"],
                )
            ],
            education=[
                Education(
                    degree="PhD Computer Science",
                    institution="MIT",
                    graduation_date="2020",
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Python Engineer",
            required_skills=["Python"],  # Not in CV
            required_experience_years=1,
            required_education="Bachelor's",
        )

        result = scorer.compute(cv_data, job_requirements)

        # Even though education and experience might score high,
        # knockout should set total to 0
        assert result.knockout_triggered is True
        assert result.total == 0.0


class TestCheckKnockout:
    """Tests for the knockout rule checking."""

    def test_no_knockout_when_all_required_skills_present(
        self,
        scorer: AlgorithmicScorer,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test no knockout when CV has all required skills."""
        knockout, reason = scorer.check_knockout(sample_cv_data, sample_job_requirements)

        assert knockout is False
        assert reason is None

    def test_knockout_when_required_skills_missing(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test knockout when required skills are missing."""
        cv_data = CVData(
            contact=ContactInfo(name="Jane Doe"),
            skills=["JavaScript", "React"],
        )

        job_requirements = JobRequirements(
            job_title="Backend Engineer",
            required_skills=["Python", "Django"],
        )

        knockout, reason = scorer.check_knockout(cv_data, job_requirements)

        assert knockout is True
        assert "Python" in reason
        assert "Django" in reason

    def test_no_knockout_with_empty_required_skills(
        self,
        scorer: AlgorithmicScorer,
        sample_cv_data: CVData,
    ) -> None:
        """Test no knockout when job has no required skills."""
        job_requirements = JobRequirements(
            job_title="General Engineer",
            required_skills=[],
            preferred_skills=["Python"],
        )

        knockout, reason = scorer.check_knockout(sample_cv_data, job_requirements)

        assert knockout is False
        assert reason is None

    def test_skill_in_experience_description_prevents_knockout(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that skills mentioned in experience text prevent knockout."""
        cv_data = CVData(
            contact=ContactInfo(name="Jane Doe"),
            skills=["JavaScript"],  # Python not in skills list
            experiences=[
                Experience(
                    title="Developer",
                    company="Corp",
                    start_date="2020",
                    end_date="Present",
                    description="Developed applications using Python and Django",
                    skills_used=[],
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Python Developer",
            required_skills=["Python"],
        )

        knockout, reason = scorer.check_knockout(cv_data, job_requirements)

        assert knockout is False

    def test_knockout_case_insensitive(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that skill matching is case-insensitive."""
        cv_data = CVData(
            contact=ContactInfo(name="Jane Doe"),
            skills=["PYTHON", "aws", "Docker"],  # Mixed case
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_skills=["python", "AWS", "docker"],  # Different case
        )

        knockout, reason = scorer.check_knockout(cv_data, job_requirements)

        assert knockout is False


class TestComputeExactMatch:
    """Tests for exact skill match computation."""

    def test_full_match_required_skills(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test score when all required skills match."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=["Python", "AWS", "Docker"],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_skills=["Python", "AWS", "Docker"],
            preferred_skills=[],
        )

        result = scorer._compute_exact_match(cv_data, job_requirements)

        # 100% required match, no preferred = 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        assert result == 1.0

    def test_partial_match_required_skills(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test score when only some required skills match."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=["Python"],  # Only 1 of 3 required
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_skills=["Python", "AWS", "Docker"],
            preferred_skills=[],
        )

        result = scorer._compute_exact_match(cv_data, job_requirements)

        # 33% required match, no preferred = 0.7 * 0.33 + 0.3 * 1.0 ≈ 0.53
        assert 0.5 < result < 0.6

    def test_match_includes_experience_skills(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that skills from experiences are included."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=["Python"],
            experiences=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date="2020",
                    skills_used=["AWS", "Docker"],
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_skills=["Python", "AWS", "Docker"],
            preferred_skills=[],
        )

        result = scorer._compute_exact_match(cv_data, job_requirements)

        # All 3 required skills found between skills list and experiences
        assert result == 1.0

    def test_match_checks_experience_text(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that skills mentioned in experience text are matched."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=[],
            experiences=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date="2020",
                    description="Worked with Python",
                    achievements=["Used AWS for deployment", "Containerized with Docker"],
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_skills=["Python", "AWS", "Docker"],
            preferred_skills=[],
        )

        result = scorer._compute_exact_match(cv_data, job_requirements)

        assert result == 1.0

    def test_no_required_skills_returns_max_for_required_portion(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that no required skills gives 1.0 for required portion."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=["Python"],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_skills=[],
            preferred_skills=["Python"],
        )

        result = scorer._compute_exact_match(cv_data, job_requirements)

        # 0.7 * 1.0 (no required) + 0.3 * 1.0 (preferred match) = 1.0
        assert result == 1.0


class TestComputeYearsFit:
    """Tests for experience years fit computation."""

    def test_exceeds_requirement(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test score when years exceed requirement."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            experiences=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date="2015",
                    end_date="2025",  # 10 years
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_experience_years=5,
        )

        result = scorer._compute_years_fit(cv_data, job_requirements)

        assert result == 1.0

    def test_exactly_meets_requirement(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test score when years exactly meet requirement."""
        current_year = datetime.now().year
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            experiences=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date=str(current_year - 5),
                    end_date="Present",
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_experience_years=5,
        )

        result = scorer._compute_years_fit(cv_data, job_requirements)

        assert result == 1.0

    def test_within_70_percent_requirement(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test score when years are 70%+ of requirement."""
        current_year = datetime.now().year
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            experiences=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date=str(current_year - 4),  # 4 years
                    end_date="Present",
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_experience_years=5,  # 4 is 80% of 5
        )

        result = scorer._compute_years_fit(cv_data, job_requirements)

        assert result == 0.8

    def test_within_50_percent_requirement(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test score when years are 50-70% of requirement."""
        current_year = datetime.now().year
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            experiences=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date=str(current_year - 3),  # 3 years
                    end_date="Present",
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_experience_years=5,  # 3 is 60% of 5
        )

        result = scorer._compute_years_fit(cv_data, job_requirements)

        assert result == 0.5

    def test_below_50_percent_requirement(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test score when years are below 50% of requirement."""
        current_year = datetime.now().year
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            experiences=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date=str(current_year - 1),  # 1 year
                    end_date="Present",
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_experience_years=10,  # 1 is 10% of 10
        )

        result = scorer._compute_years_fit(cv_data, job_requirements)

        # max(0.2, 1/10) = 0.2
        assert result == 0.2

    def test_no_requirement_returns_one(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that no years requirement returns 1.0."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            experiences=[],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_experience_years=None,
        )

        result = scorer._compute_years_fit(cv_data, job_requirements)

        assert result == 1.0

    def test_zero_requirement_returns_one(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that zero years requirement returns 1.0."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            experiences=[],
        )

        job_requirements = JobRequirements(
            job_title="Entry Level Engineer",
            required_experience_years=0,
        )

        result = scorer._compute_years_fit(cv_data, job_requirements)

        assert result == 1.0


class TestComputeEducationMatch:
    """Tests for education match computation."""

    def test_exceeds_requirement(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test score when education exceeds requirement."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            education=[
                Education(
                    degree="Master of Science",
                    institution="MIT",
                    graduation_date="2020",
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_education="Bachelor's degree",
        )

        result = scorer._compute_education_match(cv_data, job_requirements)

        assert result == 1.0

    def test_exact_match(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test score when education exactly matches."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            education=[
                Education(
                    degree="Bachelor of Science",
                    institution="State University",
                    graduation_date="2020",
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_education="Bachelor's degree",
        )

        result = scorer._compute_education_match(cv_data, job_requirements)

        assert result == 1.0

    def test_one_level_below(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test score when education is one level below requirement."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            education=[
                Education(
                    degree="Associate degree",
                    institution="Community College",
                    graduation_date="2020",
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_education="Bachelor's degree",
        )

        result = scorer._compute_education_match(cv_data, job_requirements)

        assert result == 0.7

    def test_no_requirement_returns_one(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that no education requirement returns 1.0."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            education=[],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_education=None,
        )

        result = scorer._compute_education_match(cv_data, job_requirements)

        assert result == 1.0

    def test_phd_vs_master_requirement(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test PhD candidate exceeds Master's requirement."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            education=[
                Education(
                    degree="PhD in Computer Science",
                    institution="Stanford",
                    graduation_date="2020",
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Researcher",
            required_education="Master's degree",
        )

        result = scorer._compute_education_match(cv_data, job_requirements)

        assert result == 1.0

    def test_unrecognized_requirement_returns_one(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that unrecognized education requirement returns 1.0."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            education=[
                Education(
                    degree="Bachelor of Arts",
                    institution="University",
                    graduation_date="2020",
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_education="Professional certification preferred",
        )

        result = scorer._compute_education_match(cv_data, job_requirements)

        assert result == 1.0


class TestComputeRecency:
    """Tests for recency score computation."""

    def test_no_experiences_returns_zero(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that no experiences returns 0.0."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            experiences=[],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_skills=["Python"],
        )

        result = scorer._compute_recency(cv_data, job_requirements)

        assert result == 0.0

    def test_no_target_skills_returns_one(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that no target skills returns 1.0."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            experiences=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date="2020",
                    end_date="Present",
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Engineer",
            required_skills=[],
            preferred_skills=[],
        )

        result = scorer._compute_recency(cv_data, job_requirements)

        assert result == 1.0

    def test_recent_relevant_experience_scores_high(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that recent relevant experience scores high."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            experiences=[
                Experience(
                    title="Python Engineer",
                    company="Corp",
                    start_date="2023",
                    end_date="Present",
                    description="Building Python applications",
                    skills_used=["Python", "AWS"],
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Python Developer",
            required_skills=["Python"],
            preferred_skills=["AWS"],
        )

        result = scorer._compute_recency(cv_data, job_requirements)

        assert result > 0.5

    def test_old_experience_scores_lower(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that old experience scores lower due to decay."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            experiences=[
                Experience(
                    title="Python Engineer",
                    company="Corp",
                    start_date="2010",
                    end_date="2015",  # Old
                    description="Building Python applications",
                    skills_used=["Python"],
                )
            ],
        )

        job_requirements = JobRequirements(
            job_title="Python Developer",
            required_skills=["Python"],
        )

        result = scorer._compute_recency(cv_data, job_requirements)

        # Should be lower due to ~10+ years ago decay
        assert result < 0.5


class TestExtractYear:
    """Tests for the _extract_year static method."""

    def test_extract_year_from_simple_year(self) -> None:
        """Test extracting year from simple year string."""
        result = AlgorithmicScorer._extract_year("2020")

        assert result == 2020

    def test_extract_year_from_full_date(self) -> None:
        """Test extracting year from full date string."""
        result = AlgorithmicScorer._extract_year("January 2020")

        assert result == 2020

    def test_extract_year_from_date_range(self) -> None:
        """Test extracting year from date range format."""
        result = AlgorithmicScorer._extract_year("Jan 2020 - Dec 2022")

        assert result == 2020  # Returns first match

    def test_extract_year_present(self) -> None:
        """Test that 'Present' returns current year."""
        result = AlgorithmicScorer._extract_year("Present")

        assert result == datetime.now().year

    def test_extract_year_current(self) -> None:
        """Test that 'Current' returns current year."""
        result = AlgorithmicScorer._extract_year("Current")

        assert result == datetime.now().year

    def test_extract_year_now(self) -> None:
        """Test that 'Now' returns current year."""
        result = AlgorithmicScorer._extract_year("now")

        assert result == datetime.now().year

    def test_extract_year_ongoing(self) -> None:
        """Test that 'Ongoing' returns current year."""
        result = AlgorithmicScorer._extract_year("Ongoing")

        assert result == datetime.now().year

    def test_extract_year_none_input(self) -> None:
        """Test that None input returns None."""
        result = AlgorithmicScorer._extract_year(None)

        assert result is None

    def test_extract_year_invalid_format(self) -> None:
        """Test that invalid format returns None."""
        result = AlgorithmicScorer._extract_year("No year here")

        assert result is None

    def test_extract_year_too_old(self) -> None:
        """Test that years before 1900 are not extracted."""
        result = AlgorithmicScorer._extract_year("1899")

        assert result is None


class TestCollectCVSkills:
    """Tests for _collect_cv_skills method."""

    def test_collects_from_skills_list(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test collecting skills from skills list."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=["Python", "AWS", "Docker"],
        )

        skills = scorer._collect_cv_skills(cv_data)

        assert "Python" in skills
        assert "AWS" in skills
        assert "Docker" in skills

    def test_collects_from_experience_skills(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test collecting skills from experience skills_used."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=["Python"],
            experiences=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date="2020",
                    skills_used=["AWS", "Docker"],
                )
            ],
        )

        skills = scorer._collect_cv_skills(cv_data)

        assert "Python" in skills
        assert "AWS" in skills
        assert "Docker" in skills

    def test_deduplicates_skills(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that duplicate skills are deduplicated."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            skills=["Python", "AWS"],
            experiences=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date="2020",
                    skills_used=["Python", "AWS"],  # Duplicates
                )
            ],
        )

        skills = scorer._collect_cv_skills(cv_data)

        assert isinstance(skills, set)
        assert len(skills) == 2


class TestGetExperienceText:
    """Tests for _get_experience_text method."""

    def test_concatenates_experience_fields(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that all experience fields are concatenated."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            experiences=[
                Experience(
                    title="Senior Engineer",
                    company="Tech Corp",
                    start_date="2020",
                    description="Led development",
                    achievements=["Built systems", "Led team"],
                )
            ],
        )

        text = scorer._get_experience_text(cv_data)

        assert "Senior Engineer" in text
        assert "Led development" in text
        assert "Built systems" in text
        assert "Led team" in text

    def test_empty_experiences_returns_empty(
        self,
        scorer: AlgorithmicScorer,
    ) -> None:
        """Test that no experiences returns empty string."""
        cv_data = CVData(
            contact=ContactInfo(name="John Doe"),
            experiences=[],
        )

        text = scorer._get_experience_text(cv_data)

        assert text == ""
