"""Tests for conditional edge functions in LangGraph workflow."""

from cv_warlock.graph.edges import (
    should_continue_after_extraction,
    should_continue_after_validation,
)
from cv_warlock.models.cv import ContactInfo, CVData
from cv_warlock.models.job_spec import JobRequirements


class TestShouldContinueAfterValidation:
    """Tests for should_continue_after_validation edge function."""

    def test_no_errors_returns_continue(self) -> None:
        """Test that empty errors list returns 'continue'."""
        state = {"errors": []}
        assert should_continue_after_validation(state) == "continue"

    def test_missing_errors_returns_continue(self) -> None:
        """Test that missing errors key returns 'continue'."""
        state = {}
        assert should_continue_after_validation(state) == "continue"

    def test_none_errors_returns_continue(self) -> None:
        """Test that None errors returns 'continue'."""
        state = {"errors": None}
        assert should_continue_after_validation(state) == "continue"

    def test_with_errors_returns_error(self) -> None:
        """Test that state with errors returns 'error'."""
        state = {"errors": ["Validation failed"]}
        assert should_continue_after_validation(state) == "error"

    def test_multiple_errors_returns_error(self) -> None:
        """Test that multiple errors returns 'error'."""
        state = {"errors": ["Error 1", "Error 2", "Error 3"]}
        assert should_continue_after_validation(state) == "error"


class TestShouldContinueAfterExtraction:
    """Tests for should_continue_after_extraction edge function."""

    def test_success_returns_continue(self) -> None:
        """Test that successful extraction returns 'continue'."""
        cv_data = CVData(contact=ContactInfo(name="John Doe"))
        job_requirements = JobRequirements(job_title="Software Engineer")
        state = {
            "errors": [],
            "cv_data": cv_data,
            "job_requirements": job_requirements,
        }
        assert should_continue_after_extraction(state) == "continue"

    def test_with_errors_returns_error(self) -> None:
        """Test that errors returns 'error' even with valid data."""
        cv_data = CVData(contact=ContactInfo(name="John Doe"))
        job_requirements = JobRequirements(job_title="Software Engineer")
        state = {
            "errors": ["Extraction partially failed"],
            "cv_data": cv_data,
            "job_requirements": job_requirements,
        }
        assert should_continue_after_extraction(state) == "error"

    def test_missing_cv_data_returns_error(self) -> None:
        """Test that missing cv_data returns 'error'."""
        job_requirements = JobRequirements(job_title="Software Engineer")
        state = {
            "errors": [],
            "cv_data": None,
            "job_requirements": job_requirements,
        }
        assert should_continue_after_extraction(state) == "error"

    def test_missing_job_requirements_returns_error(self) -> None:
        """Test that missing job_requirements returns 'error'."""
        cv_data = CVData(contact=ContactInfo(name="John Doe"))
        state = {
            "errors": [],
            "cv_data": cv_data,
            "job_requirements": None,
        }
        assert should_continue_after_extraction(state) == "error"

    def test_both_missing_returns_error(self) -> None:
        """Test that both missing returns 'error'."""
        state = {
            "errors": [],
            "cv_data": None,
            "job_requirements": None,
        }
        assert should_continue_after_extraction(state) == "error"

    def test_no_cv_data_key_returns_error(self) -> None:
        """Test that missing cv_data key returns 'error'."""
        job_requirements = JobRequirements(job_title="Software Engineer")
        state = {
            "errors": [],
            "job_requirements": job_requirements,
        }
        assert should_continue_after_extraction(state) == "error"

    def test_no_job_requirements_key_returns_error(self) -> None:
        """Test that missing job_requirements key returns 'error'."""
        cv_data = CVData(contact=ContactInfo(name="John Doe"))
        state = {
            "errors": [],
            "cv_data": cv_data,
        }
        assert should_continue_after_extraction(state) == "error"

    def test_empty_state_returns_error(self) -> None:
        """Test that empty state returns 'error'."""
        state = {}
        assert should_continue_after_extraction(state) == "error"
