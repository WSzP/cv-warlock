"""Tests for LangGraph state models."""

from cv_warlock.models.state import (
    CVWarlockState,
    MatchAnalysis,
    ScoreBreakdown,
    StepTiming,
    TailoringPlan,
)


class TestScoreBreakdown:
    """Tests for ScoreBreakdown TypedDict."""

    def test_create_score_breakdown(self) -> None:
        """Test creating a ScoreBreakdown."""
        breakdown: ScoreBreakdown = {
            "exact_skill_match": 0.8,
            "semantic_skill_match": 0.75,
            "document_similarity": 0.7,
            "experience_years_fit": 0.9,
            "education_match": 0.85,
            "recency_score": 0.8,
        }

        assert breakdown["exact_skill_match"] == 0.8
        assert breakdown["semantic_skill_match"] == 0.75
        assert breakdown["document_similarity"] == 0.7
        assert breakdown["experience_years_fit"] == 0.9
        assert breakdown["education_match"] == 0.85
        assert breakdown["recency_score"] == 0.8

    def test_score_values_in_range(self) -> None:
        """Test that scores should be in 0-1 range (convention, not enforced)."""
        breakdown: ScoreBreakdown = {
            "exact_skill_match": 0.0,
            "semantic_skill_match": 0.5,
            "document_similarity": 1.0,
            "experience_years_fit": 0.33,
            "education_match": 0.67,
            "recency_score": 0.99,
        }

        for key, value in breakdown.items():
            assert 0.0 <= value <= 1.0, f"{key} should be between 0 and 1"


class TestMatchAnalysis:
    """Tests for MatchAnalysis TypedDict."""

    def test_create_match_analysis(self) -> None:
        """Test creating a MatchAnalysis."""
        analysis: MatchAnalysis = {
            "strong_matches": ["Python", "AWS", "Docker"],
            "partial_matches": ["Kubernetes"],
            "gaps": ["Go", "Rust"],
            "transferable_skills": ["Leadership", "Project Management"],
            "relevance_score": 0.75,
        }

        assert len(analysis["strong_matches"]) == 3
        assert "Python" in analysis["strong_matches"]
        assert len(analysis["gaps"]) == 2
        assert analysis["relevance_score"] == 0.75

    def test_empty_match_analysis(self) -> None:
        """Test MatchAnalysis with empty lists."""
        analysis: MatchAnalysis = {
            "strong_matches": [],
            "partial_matches": [],
            "gaps": [],
            "transferable_skills": [],
            "relevance_score": 0.0,
        }

        assert analysis["strong_matches"] == []
        assert analysis["relevance_score"] == 0.0


class TestTailoringPlan:
    """Tests for TailoringPlan TypedDict."""

    def test_create_tailoring_plan(self) -> None:
        """Test creating a TailoringPlan."""
        plan: TailoringPlan = {
            "summary_focus": ["AI expertise", "Leadership experience"],
            "experiences_to_emphasize": ["Senior Engineer at Tech Corp"],
            "skills_to_highlight": ["Python", "Machine Learning"],
            "achievements_to_feature": ["Led team of 10", "Increased efficiency 50%"],
            "keywords_to_incorporate": ["microservices", "cloud-native"],
            "sections_to_reorder": ["skills", "experience", "education"],
        }

        assert len(plan["summary_focus"]) == 2
        assert "Python" in plan["skills_to_highlight"]
        assert len(plan["keywords_to_incorporate"]) == 2


class TestStepTiming:
    """Tests for StepTiming TypedDict."""

    def test_create_step_timing(self) -> None:
        """Test creating a StepTiming."""
        timing: StepTiming = {
            "step_name": "extract_cv",
            "start_time": 1705612800.0,
            "end_time": 1705612805.0,
            "duration_seconds": 5.0,
        }

        assert timing["step_name"] == "extract_cv"
        assert timing["duration_seconds"] == 5.0

    def test_step_timing_in_progress(self) -> None:
        """Test StepTiming for step in progress."""
        timing: StepTiming = {
            "step_name": "tailor_experiences",
            "start_time": 1705612800.0,
            "end_time": None,
            "duration_seconds": None,
        }

        assert timing["end_time"] is None
        assert timing["duration_seconds"] is None


class TestCVWarlockState:
    """Tests for CVWarlockState TypedDict."""

    def test_create_minimal_state(self) -> None:
        """Test creating a minimal CVWarlockState."""
        state: CVWarlockState = {
            "raw_cv": "# John Doe\nSoftware Engineer",
            "raw_job_spec": "Looking for a Python developer",
            "assume_all_tech_skills": True,
            "use_cot": True,
            "lookback_years": 4,
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

        assert state["raw_cv"] == "# John Doe\nSoftware Engineer"
        assert state["assume_all_tech_skills"] is True
        assert state["use_cot"] is True
        assert state["errors"] == []

    def test_state_with_errors(self) -> None:
        """Test CVWarlockState with errors."""
        state: CVWarlockState = {
            "raw_cv": "",
            "raw_job_spec": "",
            "assume_all_tech_skills": False,
            "use_cot": False,
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
            "current_step": "validate_inputs",
            "current_step_description": "Validating inputs",
            "errors": ["CV is empty", "Job spec is empty"],
        }

        assert len(state["errors"]) == 2
        assert "CV is empty" in state["errors"]

    def test_state_with_timings(self) -> None:
        """Test CVWarlockState with step timings."""
        timings: list[StepTiming] = [
            {
                "step_name": "validate_inputs",
                "start_time": 1705612800.0,
                "end_time": 1705612801.0,
                "duration_seconds": 1.0,
            },
            {
                "step_name": "extract_cv",
                "start_time": 1705612801.0,
                "end_time": 1705612806.0,
                "duration_seconds": 5.0,
            },
        ]

        state: CVWarlockState = {
            "raw_cv": "# CV",
            "raw_job_spec": "# Job",
            "assume_all_tech_skills": True,
            "use_cot": True,
            "lookback_years": 4,
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
            "step_timings": timings,
            "current_step_start": 1705612806.0,
            "total_generation_time": None,
            "messages": [],
            "current_step": "extract_job",
            "current_step_description": "Extracting job requirements",
            "errors": [],
        }

        assert len(state["step_timings"]) == 2
        assert state["step_timings"][0]["step_name"] == "validate_inputs"
        assert state["step_timings"][1]["duration_seconds"] == 5.0

    def test_state_cot_settings(self) -> None:
        """Test use_cot flag in state."""
        # CoT enabled
        state_cot: CVWarlockState = {
            "raw_cv": "CV",
            "raw_job_spec": "Job",
            "assume_all_tech_skills": True,
            "use_cot": True,
            "lookback_years": 4,
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
            "current_step": "init",
            "current_step_description": "Init",
            "errors": [],
        }
        assert state_cot["use_cot"] is True

        # CoT disabled
        state_no_cot = dict(state_cot)
        state_no_cot["use_cot"] = False
        assert state_no_cot["use_cot"] is False
