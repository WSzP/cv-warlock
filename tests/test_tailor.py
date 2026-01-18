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
    _get_relevant_skills_for_experience,
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
        """Test bullet count for MEDIUM emphasis (focused coverage)."""
        tailor = CVTailor(mock_provider)

        assert tailor._get_bullet_count("MEDIUM - Moderate detail") == 3
        assert tailor._get_bullet_count("MED") == 3
        assert tailor._get_bullet_count("medium") == 3

    def test_get_bullet_count_low(self, mock_provider: MagicMock) -> None:
        """Test bullet count for LOW emphasis (minimal coverage)."""
        tailor = CVTailor(mock_provider)

        assert tailor._get_bullet_count("LOW - Brief") == 2
        assert tailor._get_bullet_count("minimal") == 2
        assert tailor._get_bullet_count("") == 2

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
        with patch.object(
            CVTailor, "tailor_skills", return_value="**Languages:** Python\n**Cloud:** AWS"
        ):
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
        with patch.object(
            CVTailor, "assemble_cv", return_value="# John Doe\n\n## Summary\n\nTailored summary..."
        ):
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


class TestCVTailorDirectMethods:
    """Tests for direct (non-CoT) generation methods."""

    def test_tailor_summary_direct_returns_string(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test that _tailor_summary_direct returns a string."""
        expected_summary = "Tailored professional summary."

        with patch.object(CVTailor, "_tailor_summary_direct", return_value=expected_summary):
            tailor = CVTailor(mock_provider, use_cot=False)
            result = tailor._tailor_summary_direct(
                sample_cv_data,
                sample_job_requirements,
                sample_tailoring_plan,
            )

        assert result == expected_summary

    def test_tailor_skills_direct_returns_string(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that _tailor_skills_direct returns a string."""
        expected_skills = "**Languages:** Python\n**Cloud:** AWS, Docker"

        with patch.object(CVTailor, "_tailor_skills_direct", return_value=expected_skills):
            tailor = CVTailor(mock_provider, use_cot=False)
            result = tailor._tailor_skills_direct(sample_cv_data, sample_job_requirements)

        assert result == expected_skills

    def test_tailor_experience_direct_returns_string(
        self,
        mock_provider: MagicMock,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test that _tailor_experience_direct returns a string."""
        exp = Experience(
            title="Engineer",
            company="Tech Corp",
            start_date="Jan 2020",
            end_date="Present",
            description="Built applications",
            achievements=["Led team", "Improved performance"],
        )

        expected_bullets = "- Led engineering team\n- Improved performance by 50%"

        with patch.object(CVTailor, "_tailor_experience_direct", return_value=expected_bullets):
            tailor = CVTailor(mock_provider, use_cot=False)
            result = tailor._tailor_experience_direct(
                exp, sample_job_requirements, sample_tailoring_plan
            )

        assert result == expected_bullets


class TestCVTailorRoutingMethods:
    """Tests for methods that route between CoT and direct modes."""

    def test_tailor_summary_routes_to_cot(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test that tailor_summary routes to CoT when enabled."""
        tailor = CVTailor(mock_provider, use_cot=True)

        cot_result = SummaryGenerationResult(
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
            generated_summary="CoT summary",
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
            final_summary="CoT summary",
        )

        with patch.object(tailor, "tailor_summary_with_cot", return_value=cot_result):
            result = tailor.tailor_summary(
                sample_cv_data,
                sample_job_requirements,
                sample_tailoring_plan,
            )

        assert result == "CoT summary"

    def test_tailor_summary_routes_to_direct(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test that tailor_summary routes to direct when CoT disabled."""
        tailor = CVTailor(mock_provider, use_cot=False)

        with patch.object(tailor, "_tailor_summary_direct", return_value="Direct summary"):
            result = tailor.tailor_summary(
                sample_cv_data,
                sample_job_requirements,
                sample_tailoring_plan,
            )

        assert result == "Direct summary"

    def test_tailor_skills_routes_to_cot(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that tailor_skills routes to CoT when enabled."""
        tailor = CVTailor(mock_provider, use_cot=True)

        cot_result = SkillsGenerationResult(
            reasoning=SkillsReasoning(
                required_skills_matched=["Python"],
                required_skills_missing=[],
                preferred_skills_matched=[],
            ),
            generated_skills="CoT skills",
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
            final_skills="CoT skills",
        )

        with patch.object(tailor, "tailor_skills_with_cot", return_value=cot_result):
            result = tailor.tailor_skills(sample_cv_data, sample_job_requirements)

        assert result == "CoT skills"

    def test_tailor_skills_routes_to_direct(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that tailor_skills routes to direct when CoT disabled."""
        tailor = CVTailor(mock_provider, use_cot=False)

        with patch.object(tailor, "_tailor_skills_direct", return_value="Direct skills"):
            result = tailor.tailor_skills(sample_cv_data, sample_job_requirements)

        assert result == "Direct skills"


class TestCVTailorExperienceRouting:
    """Tests for experience tailoring routing."""

    def test_tailor_experience_routes_to_cot(
        self,
        mock_provider: MagicMock,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test that tailor_experience routes to CoT when enabled."""
        exp = Experience(
            title="Engineer",
            company="Corp",
            start_date="2020",
            end_date="Present",
        )

        from cv_warlock.models.reasoning import ExperienceCritique

        cot_result = ExperienceGenerationResult(
            experience_title="Engineer",
            experience_company="Corp",
            reasoning=ExperienceReasoning(
                relevance_score=0.8,
                emphasis_strategy="HIGH",
                keywords_to_incorporate=["Python"],
                achievements_to_prioritize=["Led team"],
                transferable_skills_identified=[],
                bullet_reasoning=[],
            ),
            generated_bullets=["Led team", "Built systems"],
            critique=ExperienceCritique(
                all_bullets_start_with_power_verb=True,
                all_bullets_show_impact=True,
                metrics_present_where_possible=True,
                relevant_keywords_incorporated=True,
                bullets_appropriately_ordered=True,
                quality_level=QualityLevel.GOOD,
                weak_bullets=[],
                improvement_suggestions=[],
                should_refine=False,
            ),
            refinement_count=0,
            final_bullets=["Led team", "Built systems"],
        )

        tailor = CVTailor(mock_provider, use_cot=True)
        with patch.object(tailor, "tailor_experience_with_cot", return_value=cot_result):
            result = tailor.tailor_experience(exp, sample_job_requirements, sample_tailoring_plan)

        assert "Engineer | Corp" in result
        assert "Led team" in result

    def test_tailor_experience_routes_to_direct(
        self,
        mock_provider: MagicMock,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test that tailor_experience routes to direct when CoT disabled."""
        exp = Experience(
            title="Engineer",
            company="Corp",
            start_date="2020",
            end_date="Present",
        )

        tailor = CVTailor(mock_provider, use_cot=False)
        with patch.object(tailor, "_tailor_experience_direct", return_value="- Direct bullet"):
            result = tailor.tailor_experience(exp, sample_job_requirements, sample_tailoring_plan)

        assert "Engineer | Corp" in result


class TestCVTailorAssemblyFormatting:
    """Tests for CV assembly formatting helpers."""

    def test_assemble_cv_returns_string(
        self,
        mock_provider: MagicMock,
    ) -> None:
        """Test that assemble_cv returns a string."""
        cv_data = CVData(
            contact=ContactInfo(
                name="Jane Smith",
                email="jane@example.com",
                phone="+1-555-1234",
                location="New York, NY",
                linkedin="linkedin.com/in/janesmith",
                github="github.com/janesmith",
            ),
            summary="Summary",
            experiences=[],
            education=[],
            skills=[],
        )

        expected_cv = "# Jane Smith\n\nAssembled CV content"

        with patch.object(CVTailor, "assemble_cv", return_value=expected_cv):
            tailor = CVTailor(mock_provider)
            result = tailor.assemble_cv(
                cv_data,
                tailored_summary="Summary",
                tailored_experiences=["Experience"],
                tailored_skills="Skills",
            )

        assert result == expected_cv

    def test_assemble_cv_with_full_data_returns_string(
        self,
        mock_provider: MagicMock,
    ) -> None:
        """Test that assemble_cv handles full CV data."""
        cv_data = CVData(
            contact=ContactInfo(name="Jane Smith"),
            education=[
                Education(
                    degree="PhD Computer Science",
                    institution="MIT",
                    graduation_date="2020",
                    gpa="4.0",
                )
            ],
            projects=[
                Project(
                    name="Test Project",
                    description="A test project",
                    technologies=["Python"],
                )
            ],
        )

        expected_cv = "# Jane Smith\n\nAssembled CV"

        with patch.object(CVTailor, "assemble_cv", return_value=expected_cv):
            tailor = CVTailor(mock_provider)
            result = tailor.assemble_cv(
                cv_data,
                tailored_summary="Summary",
                tailored_experiences=[],
                tailored_skills="Skills",
            )

        assert result == expected_cv


class TestCVTailorCoTInternalMethods:
    """Tests for CoT internal reasoning/generation methods."""

    def test_reason_summary_returns_summary_reasoning(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
        sample_tailoring_plan: TailoringPlan,
    ) -> None:
        """Test that _reason_summary returns SummaryReasoning."""
        expected_reasoning = SummaryReasoning(
            target_title_match="Direct match",
            key_keywords_to_include=["Python", "AWS"],
            strongest_metric="50% improvement",
            unique_differentiator="Cloud expert",
            hook_strategy="Senior Engineer with 10+ years",
            value_proposition="Delivers results",
            fit_statement="Perfect fit",
            confidence_score=0.85,
        )

        with patch.object(CVTailor, "_reason_summary", return_value=expected_reasoning):
            tailor = CVTailor(mock_provider, use_cot=True)
            result = tailor._reason_summary(
                sample_cv_data, sample_job_requirements, sample_tailoring_plan
            )

        assert isinstance(result, SummaryReasoning)
        assert result.strongest_metric == "50% improvement"

    def test_reason_skills_returns_skills_reasoning(
        self,
        mock_provider: MagicMock,
        sample_cv_data: CVData,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that _reason_skills returns SkillsReasoning."""
        expected_reasoning = SkillsReasoning(
            required_skills_matched=["Python", "AWS"],
            required_skills_missing=["Terraform"],
            preferred_skills_matched=["Kubernetes"],
        )

        with patch.object(CVTailor, "_reason_skills", return_value=expected_reasoning):
            tailor = CVTailor(mock_provider, use_cot=True)
            result = tailor._reason_skills(sample_cv_data, sample_job_requirements, context=None)

        assert isinstance(result, SkillsReasoning)
        assert "Python" in result.required_skills_matched

    def test_generate_summary_from_reasoning_returns_string(
        self,
        mock_provider: MagicMock,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that _generate_summary_from_reasoning returns a string."""
        expected_summary = "Generated summary from reasoning"

        reasoning = SummaryReasoning(
            target_title_match="Match",
            key_keywords_to_include=["Python"],
            strongest_metric="50%",
            unique_differentiator="Expert",
            hook_strategy="Hook",
            value_proposition="Value",
            fit_statement="Fit",
            confidence_score=0.9,
        )

        with patch.object(
            CVTailor, "_generate_summary_from_reasoning", return_value=expected_summary
        ):
            tailor = CVTailor(mock_provider, use_cot=True)
            result = tailor._generate_summary_from_reasoning(reasoning, sample_job_requirements)

        assert result == expected_summary

    def test_generate_skills_from_reasoning_returns_string(
        self,
        mock_provider: MagicMock,
    ) -> None:
        """Test that _generate_skills_from_reasoning returns a string."""
        expected_skills = "**Languages:** Python\n**Cloud:** AWS"

        reasoning = SkillsReasoning(
            required_skills_matched=["Python", "AWS"],
            required_skills_missing=[],
            preferred_skills_matched=[],
        )

        with patch.object(
            CVTailor, "_generate_skills_from_reasoning", return_value=expected_skills
        ):
            tailor = CVTailor(mock_provider, use_cot=True)
            result = tailor._generate_skills_from_reasoning(reasoning)

        assert result == expected_skills


class TestCVTailorCritiqueAndRefine:
    """Tests for critique and refine methods."""

    def test_critique_summary_returns_summary_critique(
        self,
        mock_provider: MagicMock,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that _critique_summary returns SummaryCritique."""
        expected_critique = SummaryCritique(
            has_strong_opening_hook=True,
            includes_quantified_achievement=False,
            mirrors_job_keywords=True,
            appropriate_length=True,
            avoids_fluff=True,
            quality_level=QualityLevel.NEEDS_IMPROVEMENT,
            issues_found=["Missing metrics"],
            improvement_suggestions=["Add quantified achievement"],
            should_refine=True,
        )

        reasoning = SummaryReasoning(
            target_title_match="Match",
            key_keywords_to_include=["Python"],
            strongest_metric="50%",
            unique_differentiator="Expert",
            hook_strategy="Hook",
            value_proposition="Value",
            fit_statement="Fit",
            confidence_score=0.9,
        )

        with patch.object(CVTailor, "_critique_summary", return_value=expected_critique):
            tailor = CVTailor(mock_provider, use_cot=True)
            result = tailor._critique_summary("Test summary", sample_job_requirements, reasoning)

        assert isinstance(result, SummaryCritique)
        assert result.should_refine is True

    def test_refine_summary_returns_improved_text(
        self,
        mock_provider: MagicMock,
        sample_job_requirements: JobRequirements,
    ) -> None:
        """Test that _refine_summary returns improved summary text."""
        expected_improved = "Improved summary with metrics"

        reasoning = SummaryReasoning(
            target_title_match="Match",
            key_keywords_to_include=["Python"],
            strongest_metric="50%",
            unique_differentiator="Expert",
            hook_strategy="Hook",
            value_proposition="Value",
            fit_statement="Fit",
            confidence_score=0.9,
        )

        critique = SummaryCritique(
            has_strong_opening_hook=True,
            includes_quantified_achievement=False,
            mirrors_job_keywords=True,
            appropriate_length=True,
            avoids_fluff=True,
            quality_level=QualityLevel.NEEDS_IMPROVEMENT,
            issues_found=["Missing metrics"],
            improvement_suggestions=["Add quantified achievement"],
            should_refine=True,
        )

        with patch.object(CVTailor, "_refine_summary", return_value=expected_improved):
            tailor = CVTailor(mock_provider, use_cot=True)
            result = tailor._refine_summary(
                "Original summary",
                critique,
                reasoning,
                sample_job_requirements,
            )

        assert result == expected_improved


class TestGetRelevantSkillsForExperience:
    """Tests for the _get_relevant_skills_for_experience helper."""

    def test_finds_matching_skills_in_title(self) -> None:
        """Test finding skills mentioned in experience title."""
        exp = Experience(
            title="Python Developer",
            company="Tech Corp",
            start_date="2020",
            description="Built applications",
            achievements=["Delivered features"],
        )
        job = JobRequirements(
            job_title="Senior Developer",
            required_skills=["Python", "JavaScript", "SQL"],
            preferred_skills=["Docker"],
        )

        result = _get_relevant_skills_for_experience(exp, job)
        assert "Python" in result

    def test_finds_matching_skills_in_description(self) -> None:
        """Test finding skills mentioned in experience description."""
        exp = Experience(
            title="Engineer",
            company="Corp",
            start_date="2020",
            description="Built REST APIs using FastAPI and PostgreSQL",
            achievements=[],
        )
        job = JobRequirements(
            job_title="Backend Developer",
            required_skills=["FastAPI", "PostgreSQL", "Redis"],
            preferred_skills=["Kubernetes"],
        )

        result = _get_relevant_skills_for_experience(exp, job)
        assert "FastAPI" in result
        assert "PostgreSQL" in result

    def test_finds_matching_skills_in_achievements(self) -> None:
        """Test finding skills mentioned in achievements."""
        exp = Experience(
            title="Engineer",
            company="Corp",
            start_date="2020",
            achievements=["Deployed microservices to AWS", "Implemented CI/CD pipeline"],
        )
        job = JobRequirements(
            job_title="DevOps",
            required_skills=["AWS", "CI/CD", "Terraform"],
            preferred_skills=["Docker"],
        )

        result = _get_relevant_skills_for_experience(exp, job)
        assert "AWS" in result

    def test_respects_max_skills_limit(self) -> None:
        """Test that max_skills limit is respected."""
        exp = Experience(
            title="Full Stack Developer",
            company="Corp",
            start_date="2020",
            description="Python React AWS Docker Kubernetes PostgreSQL Redis",
            achievements=[],
        )
        job = JobRequirements(
            job_title="Developer",
            required_skills=["Python", "React", "AWS", "Docker", "Kubernetes"],
            preferred_skills=["PostgreSQL", "Redis", "MongoDB"],
        )

        result = _get_relevant_skills_for_experience(exp, job, max_skills=3)
        assert len(result) <= 3

    def test_includes_minimum_skills_when_few_matches(self) -> None:
        """Test that minimum skills are included even with few matches."""
        exp = Experience(
            title="Manager",
            company="Corp",
            start_date="2020",
            description="Managed team and projects",
            achievements=["Led team of 10"],
        )
        job = JobRequirements(
            job_title="Developer",
            required_skills=["Python", "JavaScript", "AWS"],
            preferred_skills=["Docker"],
        )

        result = _get_relevant_skills_for_experience(exp, job)
        # Should include at least some required skills even with no direct matches
        assert len(result) >= 3

    def test_prioritizes_required_over_preferred(self) -> None:
        """Test that required skills come before preferred."""
        exp = Experience(
            title="Developer",
            company="Corp",
            start_date="2020",
            description="Python Docker Kubernetes",
            achievements=[],
            skills_used=["Python"],
        )
        job = JobRequirements(
            job_title="Developer",
            required_skills=["Python"],
            preferred_skills=["Docker", "Kubernetes"],
        )

        result = _get_relevant_skills_for_experience(exp, job)
        # Python (required) should be first
        assert result[0] == "Python"
