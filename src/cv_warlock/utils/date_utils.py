"""Date parsing utilities for experience filtering."""

import re
from datetime import datetime

# Pattern to match 4-digit years (1900-2099)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

# Terms indicating current/ongoing employment
PRESENT_TERMS = {"present", "current", "now", "ongoing"}


def extract_end_year(end_date: str | None) -> int | None:
    """Extract the year from an end date string.

    Args:
        end_date: End date string like "December 2021", "2021", "Present", or None.

    Returns:
        The year as integer, or None if ongoing/unparseable.

    Examples:
        >>> extract_end_year("December 2021")
        2021
        >>> extract_end_year("2021")
        2021
        >>> extract_end_year("Present")
        None
        >>> extract_end_year(None)
        None
    """
    if end_date is None:
        return None

    end_date_lower = end_date.lower().strip()

    # Check for present/current indicators
    if any(term in end_date_lower for term in PRESENT_TERMS):
        return None

    # Try to extract year using regex
    match = YEAR_PATTERN.search(end_date)
    if match:
        return int(match.group())

    return None  # Unparseable - default to tailoring


def should_tailor_experience(end_date: str | None, lookback_years: int) -> bool:
    """Determine if an experience should be tailored based on its end date.

    Args:
        end_date: The experience end date string.
        lookback_years: Number of years to look back from current year.

    Returns:
        True if the experience should be tailored, False if it should pass through.

    Logic:
        - Ongoing jobs (Present/None) -> always tailor
        - Jobs ending within lookback window -> tailor
        - Jobs ending before lookback window -> pass through unchanged
        - Unparseable dates -> tailor (safe default)
    """
    end_year = extract_end_year(end_date)

    # Ongoing or unparseable -> tailor
    if end_year is None:
        return True

    current_year = datetime.now().year
    cutoff_year = current_year - lookback_years

    # Job ended after cutoff -> tailor
    # Job ended at or before cutoff -> don't tailor
    return end_year > cutoff_year
