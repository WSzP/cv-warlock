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

    Handles both LLM-only MatchAnalysis and HybridMatchResult with score breakdown.

    Args:
        match_analysis: Match analysis results.

    Returns:
        Formatted string for display.
    """
    output = []
    score = match_analysis["relevance_score"]

    # Check if this is a hybrid result with breakdown
    if match_analysis.get("score_breakdown"):
        breakdown = match_analysis["score_breakdown"]
        knockout = match_analysis.get("knockout_triggered", False)

        output.append(f"## Match Analysis (Score: {score:.0%})")
        output.append("")

        if knockout:
            output.append(
                f"**⚠ Knockout Triggered:** {match_analysis.get('knockout_reason', 'Missing required skills')}"
            )
            output.append("")
        else:
            output.append("### Score Breakdown")
            output.append(f"- **Exact Skill Match:** {breakdown['exact_skill_match']:.0%}")
            output.append(f"- **Semantic Skill Match:** {breakdown['semantic_skill_match']:.0%}")
            output.append(f"- **Document Similarity:** {breakdown['document_similarity']:.0%}")
            output.append(f"- **Experience Years Fit:** {breakdown['experience_years_fit']:.0%}")
            output.append(f"- **Education Match:** {breakdown['education_match']:.0%}")
            output.append(f"- **Recency Score:** {breakdown['recency_score']:.0%}")
            output.append("")

            algo_score = match_analysis.get("algorithmic_score", 0)
            llm_adj = match_analysis.get("llm_adjustment", 0)
            if llm_adj != 0:
                adj_sign = "+" if llm_adj > 0 else ""
                output.append(
                    f"*Algorithmic: {algo_score:.0%}, LLM adjustment: {adj_sign}{llm_adj:.0%}*"
                )
                output.append("")
    else:
        output.append(f"## Match Analysis (Score: {score:.0%})")
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
        return "Errors occurred:\n" + "\n".join(f"- {e}" for e in state["errors"])

    output = []

    if state.get("match_analysis"):
        output.append(format_match_analysis(state["match_analysis"]))
        output.append("\n---\n")

    if state.get("tailored_cv"):
        output.append("## Tailored CV\n")
        output.append(state["tailored_cv"])

    return "\n".join(output)
