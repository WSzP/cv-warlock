"""Result display component for Streamlit UI."""

from typing import Any

import streamlit as st


def render_result(result: dict[str, Any]) -> None:
    """Render the tailoring result.

    Args:
        result: The workflow result state.
    """
    st.divider()

    # Check for errors
    if result.get("errors"):
        st.error("Errors occurred during processing:")
        for error in result["errors"]:
            st.write(f"- {error}")
        return

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Tailored CV", "Match Analysis", "Tailoring Plan"])

    with tab1:
        render_tailored_cv(result)

    with tab2:
        render_match_analysis(result)

    with tab3:
        render_tailoring_plan(result)


def render_tailored_cv(result: dict[str, Any]) -> None:
    """Render the tailored CV tab."""
    if not result.get("tailored_cv"):
        st.warning("No tailored CV was generated.")
        return

    st.subheader("Your Tailored CV")

    # Display the CV
    st.markdown(result["tailored_cv"])

    # Download button
    st.download_button(
        label="Download as Markdown",
        data=result["tailored_cv"],
        file_name="tailored_cv.md",
        mime="text/markdown",
    )


def render_match_analysis(result: dict[str, Any]) -> None:
    """Render the match analysis tab."""
    if not result.get("match_analysis"):
        st.warning("No match analysis available.")
        return

    analysis = result["match_analysis"]

    # Score display
    score = analysis["relevance_score"]
    score_color = "green" if score >= 0.7 else "orange" if score >= 0.5 else "red"

    col1, col2, col3 = st.columns(3)
    with col2:
        st.metric(
            "Match Score",
            f"{score:.0%}",
            delta=None,
        )

    st.divider()

    # Analysis sections
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Strong Matches")
        for item in analysis["strong_matches"]:
            st.markdown(f"- {item}")

        st.subheader("Transferable Skills")
        for item in analysis["transferable_skills"]:
            st.markdown(f"- {item}")

    with col2:
        st.subheader("Partial Matches")
        for item in analysis["partial_matches"]:
            st.markdown(f"- {item}")

        st.subheader("Gaps")
        for item in analysis["gaps"]:
            st.markdown(f"- {item}")


def render_tailoring_plan(result: dict[str, Any]) -> None:
    """Render the tailoring plan tab."""
    if not result.get("tailoring_plan"):
        st.warning("No tailoring plan available.")
        return

    plan = result["tailoring_plan"]

    st.subheader("Summary Focus")
    for item in plan["summary_focus"]:
        st.markdown(f"- {item}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Skills to Highlight")
        for item in plan["skills_to_highlight"]:
            st.markdown(f"- {item}")

        st.subheader("Achievements to Feature")
        for item in plan["achievements_to_feature"]:
            st.markdown(f"- {item}")

    with col2:
        st.subheader("Experiences to Emphasize")
        for item in plan["experiences_to_emphasize"]:
            st.markdown(f"- {item}")

        st.subheader("Keywords to Incorporate")
        for item in plan["keywords_to_incorporate"]:
            st.markdown(f"- {item}")
