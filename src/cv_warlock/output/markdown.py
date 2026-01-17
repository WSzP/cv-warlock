"""Markdown output formatting."""

from pathlib import Path

from cv_warlock.models.state import CVWarlockState, MatchAnalysis


def save_markdown(content: str, output_path: str | Path) -> Path:
    """Save content to a markdown file.

    Args:
        content: Markdown content to save.
        output_path: Path to save the file.

    Returns:
        Path to the saved file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def format_match_analysis(match_analysis: MatchAnalysis) -> str:
    """Format match analysis for display.

    Args:
        match_analysis: Match analysis results.

    Returns:
        Formatted string for display.
    """
    output = []
    output.append(f"## Match Analysis (Score: {match_analysis['relevance_score']:.0%})")
    output.append("")

    output.append("### Strong Matches")
    for item in match_analysis["strong_matches"]:
        output.append(f"- {item}")
    output.append("")

    output.append("### Partial Matches")
    for item in match_analysis["partial_matches"]:
        output.append(f"- {item}")
    output.append("")

    output.append("### Gaps")
    for item in match_analysis["gaps"]:
        output.append(f"- {item}")
    output.append("")

    output.append("### Transferable Skills")
    for item in match_analysis["transferable_skills"]:
        output.append(f"- {item}")

    return "\n".join(output)


def format_result(state: CVWarlockState) -> str:
    """Format the complete result for display.

    Args:
        state: Final workflow state.

    Returns:
        Formatted result string.
    """
    if state.get("errors"):
        return f"Errors occurred:\n" + "\n".join(f"- {e}" for e in state["errors"])

    output = []

    if state.get("match_analysis"):
        output.append(format_match_analysis(state["match_analysis"]))
        output.append("\n---\n")

    if state.get("tailored_cv"):
        output.append("## Tailored CV\n")
        output.append(state["tailored_cv"])

    return "\n".join(output)
