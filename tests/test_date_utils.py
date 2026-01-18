"""Tests for date parsing utilities."""

from datetime import datetime
from unittest.mock import patch

import pytest

from cv_warlock.utils.date_utils import (
    PRESENT_TERMS,
    YEAR_PATTERN,
    extract_end_year,
    should_tailor_experience,
)


class TestYearPattern:
    """Tests for the YEAR_PATTERN regex."""

    def test_matches_four_digit_years(self) -> None:
        assert YEAR_PATTERN.search("2021") is not None
        assert YEAR_PATTERN.search("2024") is not None
        assert YEAR_PATTERN.search("1999") is not None

    def test_matches_years_in_text(self) -> None:
        match = YEAR_PATTERN.search("December 2021")
        assert match is not None
        assert match.group() == "2021"

    def test_does_not_match_invalid_years(self) -> None:
        assert YEAR_PATTERN.search("999") is None
        assert YEAR_PATTERN.search("22021") is None


class TestPresentTerms:
    """Tests for PRESENT_TERMS constant."""

    def test_contains_expected_terms(self) -> None:
        assert "present" in PRESENT_TERMS
        assert "current" in PRESENT_TERMS
        assert "now" in PRESENT_TERMS
        assert "ongoing" in PRESENT_TERMS


class TestExtractEndYear:
    """Tests for extract_end_year function."""

    def test_none_returns_none(self) -> None:
        assert extract_end_year(None) is None

    def test_present_returns_none(self) -> None:
        assert extract_end_year("Present") is None
        assert extract_end_year("present") is None
        assert extract_end_year("PRESENT") is None

    def test_current_returns_none(self) -> None:
        assert extract_end_year("Current") is None
        assert extract_end_year("current") is None

    def test_now_returns_none(self) -> None:
        assert extract_end_year("Now") is None
        assert extract_end_year("now") is None

    def test_ongoing_returns_none(self) -> None:
        assert extract_end_year("Ongoing") is None
        assert extract_end_year("ongoing") is None

    def test_year_only(self) -> None:
        assert extract_end_year("2021") == 2021
        assert extract_end_year("2024") == 2024

    def test_month_year(self) -> None:
        assert extract_end_year("December 2021") == 2021
        assert extract_end_year("January 2020") == 2020
        assert extract_end_year("May 2019") == 2019

    def test_date_with_whitespace(self) -> None:
        assert extract_end_year("  2021  ") == 2021
        assert extract_end_year("  December 2021  ") == 2021

    def test_present_with_whitespace(self) -> None:
        assert extract_end_year("  Present  ") is None
        assert extract_end_year("  present  ") is None

    def test_unparseable_returns_none(self) -> None:
        # No year found - returns None
        assert extract_end_year("Unknown") is None
        assert extract_end_year("TBD") is None


class TestShouldTailorExperience:
    """Tests for should_tailor_experience function."""

    @patch("cv_warlock.utils.date_utils.datetime")
    def test_present_always_tailored(self, mock_datetime: datetime) -> None:
        mock_datetime.now.return_value.year = 2026
        assert should_tailor_experience("Present", lookback_years=4) is True
        assert should_tailor_experience("present", lookback_years=4) is True
        assert should_tailor_experience("Current", lookback_years=4) is True

    @patch("cv_warlock.utils.date_utils.datetime")
    def test_none_always_tailored(self, mock_datetime: datetime) -> None:
        mock_datetime.now.return_value.year = 2026
        assert should_tailor_experience(None, lookback_years=4) is True

    @patch("cv_warlock.utils.date_utils.datetime")
    def test_recent_experience_tailored(self, mock_datetime: datetime) -> None:
        mock_datetime.now.return_value.year = 2026
        # 2026 - 4 = 2022 cutoff, so 2023+ should be tailored
        assert should_tailor_experience("2023", lookback_years=4) is True
        assert should_tailor_experience("2024", lookback_years=4) is True
        assert should_tailor_experience("2025", lookback_years=4) is True
        assert should_tailor_experience("December 2023", lookback_years=4) is True

    @patch("cv_warlock.utils.date_utils.datetime")
    def test_old_experience_not_tailored(self, mock_datetime: datetime) -> None:
        mock_datetime.now.return_value.year = 2026
        # 2026 - 4 = 2022 cutoff, so 2022 and earlier should NOT be tailored
        assert should_tailor_experience("2022", lookback_years=4) is False
        assert should_tailor_experience("2021", lookback_years=4) is False
        assert should_tailor_experience("2020", lookback_years=4) is False
        assert should_tailor_experience("December 2021", lookback_years=4) is False

    @patch("cv_warlock.utils.date_utils.datetime")
    def test_cutoff_boundary(self, mock_datetime: datetime) -> None:
        mock_datetime.now.return_value.year = 2026
        # At exactly the cutoff year, should NOT tailor
        assert should_tailor_experience("2022", lookback_years=4) is False
        # One year after cutoff, should tailor
        assert should_tailor_experience("2023", lookback_years=4) is True

    @patch("cv_warlock.utils.date_utils.datetime")
    def test_different_lookback_years(self, mock_datetime: datetime) -> None:
        mock_datetime.now.return_value.year = 2026

        # lookback_years=0: only current year
        assert should_tailor_experience("2025", lookback_years=0) is False
        assert should_tailor_experience("2026", lookback_years=0) is False  # At cutoff
        assert should_tailor_experience("Present", lookback_years=0) is True

        # lookback_years=10: 2026 - 10 = 2016 cutoff
        assert should_tailor_experience("2015", lookback_years=10) is False
        assert should_tailor_experience("2016", lookback_years=10) is False
        assert should_tailor_experience("2017", lookback_years=10) is True

    @patch("cv_warlock.utils.date_utils.datetime")
    def test_unparseable_date_tailored(self, mock_datetime: datetime) -> None:
        """Unparseable dates default to tailoring (safe default)."""
        mock_datetime.now.return_value.year = 2026
        assert should_tailor_experience("Unknown", lookback_years=4) is True
        assert should_tailor_experience("TBD", lookback_years=4) is True
