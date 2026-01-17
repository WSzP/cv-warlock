"""Conditional edge functions for LangGraph workflow."""

from typing import Literal

from cv_warlock.models.state import CVWarlockState


def should_continue_after_validation(
    state: CVWarlockState,
) -> Literal["continue", "error"]:
    """Check if validation passed.

    Args:
        state: Current workflow state.

    Returns:
        "continue" if no errors, "error" otherwise.
    """
    if state.get("errors") and len(state["errors"]) > 0:
        return "error"
    return "continue"


def should_continue_after_extraction(
    state: CVWarlockState,
) -> Literal["continue", "error"]:
    """Check if extraction was successful.

    Args:
        state: Current workflow state.

    Returns:
        "continue" if extraction succeeded, "error" otherwise.
    """
    if state.get("errors") and len(state["errors"]) > 0:
        return "error"

    if state.get("cv_data") is None or state.get("job_requirements") is None:
        return "error"

    return "continue"
