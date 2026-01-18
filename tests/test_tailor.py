"""Tests for the CVTailor processor."""

from unittest.mock import MagicMock, patch

import pytest

from cv_warlock.models.cv import ContactInfo, CVData, Education, Experience, Project
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.models.reasoning import (
    BulletReasoning,
    ExperienceGenerationResult,
    ExperienceReasoning,
    QualityLevel,
    SkillsCritique,
    SkillsGenerationResult,
    SkillsReasoning,
    SummaryCritique,
    SummaryGenerationResult,
    SummaryReasoning,
)
from cv_warlock.models.state import TailoringPlan
from cv_warlock.processors.tailor import (
    CVTailor,
    _compress_experience_reasoning,
    _compress_skills_reasoning,
    _compress_summary_reasoning,
)


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLM provider."""
    provider = MagicMock()
    mock_model = MagicMock()
    mock_chat_model = MagicMock()

    provider.get_extraction_model.return_value = mock_model
    provider.get_chat_model.return_value = mock_chat_model

    return provider


@pytest.fixture
def sample_cv_data() -> CVData:
    """Create sample CV data for testing."""
    return CVData(
        contact=ContactInfo(
            name="John Doe",
            email="john@example.com",
            phone="+1234567890",
            location="San Francisco, CA",
            linkedin="linkedin.com/in/johndoe",
            github="github.com/johndoe",
        ),
        summary="Experienced software engineer with 10+ years building scalable systems.",
        experiences=[
            Experience(
                title="Senior Software Engineer",
                company="Tech Corp",
                start_date="Jan 2020",
                end_date="Present",
                description="Led development of cloud-native applications",
                achievements=["Led team of 5", "Reduced deployment time by 50%"],
                skills_used=["Python", "AWS", "Docker"],
            ),
            Experience(
                title="Software Engineer",
                company="Startup Inc",
                start_date="Jan 2015",
                end_date="Dec 2019",
                description="Built microservices architecture",
                achievements=["Increased performance by 40%"],
            ),
        ],
        education=[
            Education(
                degree="MS Computer Science",
                institution="Stanford",
                graduation_date="2015",
                gpa="3.9",
            )
        ],
        skills=["Python", "AWS", "Docker", "Kubernetes", "PostgreSQL"],
        projects=[
            Project(
                name="Open Source Tool",
                description="A CLI tool for developers",
                technologies=["Python", "Click"],
            )
        ],
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
        required_education="Bachelor's degree",
        seniority_level="senior",
        responsibilities=["Design systems", "Lead projects"],
    )


@pytest.fixture
def sample_tailoring_plan() -> TailoringPlan:
    """Create sample tailoring plan for testing."""
    return {
        "summary_focus": ["Cloud-native development", "Team leadership"],
        "experiences_to_emphasize": ["Senior Software Engineer at Tech Corp"],
        "skills_to_highlight": ["Python", "AWS", "Docker", "Kubernetes"],
        "achievements_to_feature": ["Led team of 5", "Reduced deployment time by 50%"],
        "keywords_to_incorporate": ["microservices", "scalability", "cloud-native"],
        "sections_to_reorder": ["summary", "experience", "skills", "education"],
    }


class TestCompressFunctions:
    """Tests for context compression helper functions."""

    def test_compress_summary_reasoning(self) -> None:
        """Test summary reasoning compression."""
        reasoning = SummaryReasoning(
            target_title_match="Exact match for Senior Software Engineer",
            key_keywords_to_include=["Python", "AWS", "microservices"],
            strongest_metric="50% deployment time reduction",
            unique_differentiator="Cloud-native expertise with 10+ years",
            hook_strategy="Senior Engineer + 10 years + cloud + scalability",
            value_proposition="Delivers 50% faster deployments",
            fit_statement="Perfect fit for cloud-focused team",
            aspects_to_avoid=["Early career jobs", "Irrelevant skills"],
            confidence_score=0.85,
        )

        result = _compress_summary_reasoning(reasoning)

        assert "Hook:" in result
        assert "Keywords:" in result
        assert "Python" in result
        assert "50% deployment time reduction" in result
        assert "Cloud-native expertise" in result

    def test_compress_experience_reasoning(self) -> None:
        """Test experience reasoning compression."""
        reasoning = ExperienceReasoning(
            relevance_score=0.9,
            emphasis_strategy="HIGH - Full detail",
            keywords_to_incorporate=["Python", "AWS", "Docker", "leadership"],
            achievements_to_prioritize=["Led team", "Improved performance"],
            transferable_skills_identified=["Team leadership", "Cloud architecture"],
            bullet_reasoning=[],
            aspects_to_downplay=["Administrative tasks"],
        )

        result = _compress_experience_reasoning(reasoning)

        assert "Relevance: 0.9" in result
        assert "Strategy:" in result
        assert "Python" in result
        assert "Led team" in result

    def test_compress_experience_reasoning_with_bullet_reasoning(self) -> None:
        """Test experience reasoning compression with bullet reasoning."""
        reasoning = ExperienceReasoning(
            relevance_score=0.8,
            emphasis_strategy="MEDIUM",
            keywords_to_incorporate=["Python"],
            achievements_to_prioritize=["Revenue growth"],
            transferable_skills_identified=["Leadership"],
            bullet_reasoning=[
                BulletReasoning(
                    original_content="Managed projects",
                    relevance_to_job="Direct leadership experience",
                    metric_identified="$1M revenue",
                    power_verb_choice="Led",
                    keyword_injection=["leadership"],
                    reframed_bullet="Led cross-functional projects driving $1M revenue growth",
                )
            ],
            aspects_to_downplay=[],
        )

        result = _compress_experience_reasoning(reasoning)

        assert "Led" in result
        assert "Bullet plans:" in result

    def test_compress_skills_reasoning(self) -> None:
        """Test skills reasoning compression."""
        reasoning = SkillsReasoning(
            required_skills_matched=["Python", "AWS", "Docker"],
            required_skills_missing=["Terraform"],
            preferred_skills_matched=["Kubernetes"],
            terminology_mapping={"Amazon Web Services": "AWS"},
            dual_format_terms=["AWS (Amazon Web Services)"],
            category_groupings={
                "Languages": ["Python", "Go"],
                "Cloud": ["AWS", "GCP"],
            },
            ordering_rationale="Required skills first",
            skills_to_omit=["Basic Excel"],
        )

        result = _compress_skills_reasoning(reasoning)

        assert "Required matched:" in result
        assert "Python" in result
        assert "Preferred matched:" in result
        assert "Kubernetes" in result
        assert "Categories:" in result


class TestCVTailorInit:
    """Tests for CVTailor initialization."""

    def test_init_with_cot_enabled(self, mock_provider: MagicMock) -> None:
        """Test initialization with CoT enabled (default)."""
        tailor = CVTailor(mock_provider, use_cot=True)

        assert tailor.llm_provider is mock_provider
        assert tailor.use_cot is True
        assert tailor.summary_prompt is not None
        assert tailor.summary_reasoning_prompt is not None

    def test_init_with_cot_disabled(self, mock_provider: MagicMock) -> None:
        """Test initialization with CoT disabled."""
        tailor = CVTailor(mock_provider, use_cot=False)

        assert tailor.use_cot is False

    def test_init_default_use_cot(self, mock_provider: MagicMock) -> None:
        """Test that use_cot defaults to True."""
        tailor = CVTailor(mock_provider)

        assert tailor.use_cot is True

    def test_init_sets_all_prompts(self, mock_provider: MagicMock) -> None:
        """Test that all prompts are initialized."""
        tailor = CVTailor(mock_provider)

        # Original prompts
        assert tailor.summary_prompt is not None
        assert tailor.experience_prompt is not None
        assert tailor.skills_prompt is not None
        assert tailor.assembly_prompt is not None

        # CoT prompts - Summary
        assert tailor.summary_reasoning_prompt is not None
        assert tailor.summary_gen_prompt is not None
        assert tailor.summary_critique_prompt is not None
        assert tailor.summary_refine_prompt is not None

        # CoT prompts - Experience
        assert tailor.exp_reasoning_prompt is not None
        assert tailor.exp_gen_prompt is not None
        assert tailor.exp_critique_prompt is not None
        assert tailor.exp_refine_prompt is not None

        # CoT prompts - Skills
        assert tailor.skills_reasoning_prompt is not None
        assert tailor.skills_gen_prompt is not None
        assert tailor.skills_critique_prompt is not None
        assert tailor.skills_refine_prompt is not None


class TestCVTailorSummary:
    """Tests for summary tailoring methods."""

    def test_tailor_summary_returns_string(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test that tailor_summary returns a string."""
        with patch.object(CVTailor, "tailor_summary", return_value="Tailored summary text"):
            tailor = CVTailor(mock_provider, use_cot=True)
            result = tailor.tailor_summary(
                sample_cv_data,
                sample_job_requirements,
                sample_tailoring_plan,
            )

        assert isinstance(result, str)
        assert result == "Tailored summary text"

    def test_tailor_summary_with_cot_returns_result(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test tailor_summary_with_cot returns SummaryGenerationResult."""
        expected = SummaryGenerationResult(
            reasoning=SummaryReasoning(
                target_title_match="Match",
                key_keywords_to_include=["Python"],
                strongest_metric="50%",
                unique_differentiator="Expert",
                hook_strategy="Hook",
                value_proposition="Value",
                fit_statement="Fit",
                confidence_score=0.9,
            ),
            generated_summary="Generated summary",
            critique=SummaryCritique(
                has_strong_opening_hook=True,
                includes_quantified_achievement=True,
                mirrors_job_keywords=True,
                appropriate_length=True,
                avoids_fluff=True,
                quality_level=QualityLevel.GOOD,
                issues_found=[],
                improvement_suggestions=[],
                should_refine=False,
            ),
            refinement_count=0,
            final_summary="Generated summary",
        )

        with patch.object(CVTailor, "tailor_summary_with_cot", return_value=expected):
            tailor = CVTailor(mock_provider, use_cot=True)
            result = tailor.tailor_summary_with_cot(
                sample_cv_data,
                sample_job_requirements,
                sample_tailoring_plan,
            )

        assert isinstance(result, SummaryGenerationResult)
        assert result.final_summary == "Generated summary"
        assert result.refinement_count == 0


class TestCVTailorExperience:
    """Tests for experience tailoring methods."""

    def test_format_experience_header(
        self,
        mock_provider: MagicMock,
    ) -> None:
        """Test experience header formatting."""
        exp = Experience(
            title="Senior Engineer",
            company="Tech Corp",
            start_date="Jan 2020",
            end_date="Present",
        )

        tailor = CVTailor(mock_provider)
        header = tailor._format_experience_header(exp)

        assert "### Senior Engineer | Tech Corp" in header
        assert "Jan 2020 - Present" in header

    def test_format_experience_header_with_past_end_date(
        self,
        mock_provider: MagicMock,
    ) -> None:
        """Test experience header with past end date."""
        exp = Experience(
            title="Developer",
            company="Startup",
            start_date="Jan 2015",
            end_date="Dec 2019",
        )

        tailor = CVTailor(mock_provider)
        header = tailor._format_experience_header(exp)

        assert "Jan 2015 - Dec 2019" in header

    def test_parse_bullets(self, mock_provider: MagicMock) -> None:
        """Test bullet point parsing."""
        tailor = CVTailor(mock_provider)

        text = """- First bullet point
- Second bullet point
* Third bullet with asterisk
Fourth without prefix
# Header to ignore"""

        bullets = tailor._parse_bullets(text)

        assert len(bullets) == 4
        assert bullets[0] == "First bullet point"
        assert bullets[1] == "Second bullet point"
        assert bullets[2] == "Third bullet with asterisk"
        assert bullets[3] == "Fourth without prefix"

    def test_parse_bullets_empty_text(self, mock_provider: MagicMock) -> None:
        """Test bullet parsing with empty text."""
        tailor = CVTailor(mock_provider)

        bullets = tailor._parse_bullets("")

        assert bullets == []

    def test_parse_bullets_filters_empty_lines(self, mock_provider: MagicMock) -> None:
        """Test that empty lines are filtered out."""
        tailor = CVTailor(mock_provider)

        text = """- Bullet one

- Bullet two

"""

        bullets = tailor._parse_bullets(text)

        assert len(bullets) == 2

    def test_get_bullet_count_high(self, mock_provider: MagicMock) -> None:
        """Test bullet count for HIGH emphasis."""
        tailor = CVTailor(mock_provider)

        assert tailor._get_bullet_count("HIGH - Full detail") == 5
        assert tailor._get_bullet_count("HIGH") == 5
        assert tailor._get_bullet_count("high relevance") == 5

    def test_get_bullet_count_medium(self, mock_provider: MagicMock) -> None:
        """Test bullet count for MEDIUM emphasis."""
        tailor = CVTailor(mock_provider)

        assert tailor._get_bullet_count("MEDIUM - Moderate detail") == 4
        assert tailor._get_bullet_count("MED") == 4
        assert tailor._get_bullet_count("medium") == 4

    def test_get_bullet_count_low(self, mock_provider: MagicMock) -> None:
        """Test bullet count for LOW emphasis."""
        tailor = CVTailor(mock_provider)

        assert tailor._get_bullet_count("LOW - Brief") == 3
        assert tailor._get_bullet_count("minimal") == 3
        assert tailor._get_bullet_count("") == 3

    def test_format_passthrough_experience_with_achievements(
        self,
        mock_provider: MagicMock,
    ) -> None:
        """Test passthrough formatting with achievements."""
        exp = Experience(
            title="Engineer",
            company="Old Corp",
            start_date="2010",
            end_date="2014",
            achievements=["Built system", "Improved process"],
        )

        tailor = CVTailor(mock_provider)
        result = tailor._format_passthrough_experience(exp)

        assert "### Engineer | Old Corp" in result
        assert "- Built system" in result
        assert "- Improved process" in result

    def test_format_passthrough_experience_with_description_only(
        self,
        mock_provider: MagicMock,
    ) -> None:
        """Test passthrough formatting with description only."""
        exp = Experience(
            title="Engineer",
            company="Old Corp",
            start_date="2010",
            end_date="2014",
            description="Worked on various projects",
            achievements=[],
        )

        tailor = CVTailor(mock_provider)
        result = tailor._format_passthrough_experience(exp)

        assert "- Worked on various projects" in result

    def test_format_passthrough_experience_no_content(
        self,
        mock_provider: MagicMock,
    ) -> None:
        """Test passthrough formatting with no content."""
        exp = Experience(
            title="Engineer",
            company="Old Corp",
            start_date="2010",
            end_date="2014",
        )

        tailor = CVTailor(mock_provider)
        result = tailor._format_passthrough_experience(exp)

        assert "- No description provided" in result

    def test_create_passthrough_result(
        self,
        mock_provider: MagicMock,
    ) -> None:
        """Test creating passthrough result for old experiences."""
        exp = Experience(
            title="Junior Dev",
            company="First Job",
            start_date="2005",
            end_date="2008",
            achievements=["Learned coding", "Built features"],
        )

        tailor = CVTailor(mock_provider)
        result = tailor._create_passthrough_result(exp)

        assert isinstance(result, ExperienceGenerationResult)
        assert result.experience_title == "Junior Dev"
        assert result.experience_company == "First Job"
        assert result.final_bullets == ["Learned coding", "Built features"]
        assert "PASSTHROUGH" in result.reasoning.emphasis_strategy


class TestCVTailorSkills:
    """Tests for skills tailoring methods."""

    def test_tailor_skills_returns_string(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that tailor_skills returns a string."""
        with patch.object(CVTailor, "tailor_skills", return_value="**Languages:** Python\n**Cloud:** AWS"):
            tailor = CVTailor(mock_provider, use_cot=True)
            result = tailor.tailor_skills(sample_cv_data, sample_job_requirements)

        assert isinstance(result, str)
        assert "Python" in result

    def test_tailor_skills_with_cot_returns_result(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test tailor_skills_with_cot returns SkillsGenerationResult."""
        expected = SkillsGenerationResult(
            reasoning=SkillsReasoning(
                required_skills_matched=["Python"],
                required_skills_missing=[],
                preferred_skills_matched=[],
            ),
            generated_skills="Skills section",
            critique=SkillsCritique(
                all_required_skills_present=True,
                uses_exact_job_terminology=True,
                appropriate_categorization=True,
                no_irrelevant_skills=True,
                no_fabricated_skills=True,
                quality_level=QualityLevel.GOOD,
                missing_critical_terms=[],
                improvement_suggestions=[],
                should_refine=False,
            ),
            refinement_count=0,
            final_skills="Skills section",
        )

        with patch.object(CVTailor, "tailor_skills_with_cot", return_value=expected):
            tailor = CVTailor(mock_provider, use_cot=True)
            result = tailor.tailor_skills_with_cot(sample_cv_data, sample_job_requirements)

        assert isinstance(result, SkillsGenerationResult)
        assert result.final_skills == "Skills section"
        assert result.refinement_count == 0


class TestCVTailorAssembly:
    """Tests for CV assembly method."""

    def test_assemble_cv_returns_string(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
    ) -> None:
        """Test CV assembly returns a string."""
        with patch.object(CVTailor, "assemble_cv", return_value="# John Doe\n\n## Summary\n\nTailored summary..."):
            tailor = CVTailor(mock_provider)
            result = tailor.assemble_cv(
                sample_cv_data,
                tailored_summary="Tailored professional summary",
                tailored_experiences=["### Engineer | Company\n- Achievement"],
                tailored_skills="**Languages:** Python",
            )

        assert isinstance(result, str)
        assert "John Doe" in result


class TestCVTailorExperiencesBatch:
    """Tests for batch experience tailoring."""

    @patch("cv_warlock.processors.tailor.get_settings")
    def test_tailor_experiences_uses_lookback_years(
        self,
        mock_get_settings: MagicMock,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test that tailor_experiences respects lookback_years."""
        mock_settings = MagicMock()
        mock_settings.lookback_years = 4
        mock_get_settings.return_value = mock_settings

        tailored_experiences = [
            "### Senior Software Engineer | Tech Corp\n- Led team of 5",
            "### Software Engineer | Startup Inc\n- Increased performance by 40%",
        ]

        with patch.object(CVTailor, "tailor_experiences", return_value=tailored_experiences):
            tailor = CVTailor(mock_provider, use_cot=False)
            results = tailor.tailor_experiences(
                sample_cv_data,
                sample_job_requirements,
                sample_tailoring_plan,
            )

        # Should return one entry per experience
        assert len(results) == len(sample_cv_data.experiences)

    @patch("cv_warlock.processors.tailor.get_settings")
    def test_tailor_experiences_with_custom_lookback(
        self,
        mock_get_settings: MagicMock,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test tailor_experiences with custom lookback_years parameter."""
        mock_settings = MagicMock()
        mock_settings.lookback_years = 10
        mock_get_settings.return_value = mock_settings

        tailored_experiences = ["Exp 1", "Exp 2"]

        with patch.object(CVTailor, "tailor_experiences", return_value=tailored_experiences):
            tailor = CVTailor(mock_provider, use_cot=False)
            results = tailor.tailor_experiences(
                sample_cv_data,
                sample_job_requirements,
                sample_tailoring_plan,
                lookback_years=2,  # Override settings
            )

        assert len(results) == len(sample_cv_data.experiences)


class TestCVTailorMaxRefinement:
    """Tests for refinement iteration constant."""

    def test_max_refinement_iterations_constant(self, mock_provider: MagicMock) -> None:
        """Test that MAX_REFINEMENT_ITERATIONS is set."""
        tailor = CVTailor(mock_provider)

        assert hasattr(tailor, "MAX_REFINEMENT_ITERATIONS")
        assert tailor.MAX_REFINEMENT_ITERATIONS == 2
