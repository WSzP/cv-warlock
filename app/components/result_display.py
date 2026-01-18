"""Result display component for Streamlit UI."""

import re
from typing import Any

import streamlit as st

from app.utils.pdf_generator import generate_cv_pdf
from cv_warlock.models.reasoning import CoverLetterGenerationResult


def _run_retest_scoring(edited_cv: str, result: dict[str, Any]) -> dict[str, Any] | None:
    """Re-run scoring on edited CV content.

    Args:
        edited_cv: The edited CV markdown content.
        result: Original result containing cv_data and job_requirements.

    Returns:
        Updated match analysis dict, or None if scoring fails.
    """
    try:
        from cv_warlock.extractors.cv_extractor import CVExtractor
        from cv_warlock.llm.base import get_llm_provider
        from cv_warlock.scoring.hybrid import HybridScorer

        # Get the process params from session state
        params = st.session_state.get("process_params", {})
        if not params:
            st.error("Cannot retest: missing processing parameters. Please generate a new CV.")
            return None

        # Create LLM provider
        llm_provider = get_llm_provider(
            provider=params.get("provider", "anthropic"),
            model=params.get("model", "claude-sonnet-4-5-20250929"),
            api_key=params.get("api_key"),
        )

        # Re-extract CV data from edited content
        cv_extractor = CVExtractor(llm_provider)
        cv_data = cv_extractor.extract(edited_cv)

        # Get original job requirements (these don't change)
        job_requirements = result.get("job_requirements")
        if not job_requirements:
            st.error("Cannot retest: missing job requirements.")
            return None

        # Run hybrid scoring
        hybrid_scorer = HybridScorer(llm_provider)
        match_analysis = hybrid_scorer.score(cv_data, job_requirements)

        return match_analysis

    except Exception as e:
        st.error(f"Retest failed: {e}")
        return None


def _render_match_score_card(result: dict[str, Any]) -> None:
    """Render the match score card with edit tracking and retest button.

    Shows:
    - Match score prominently
    - Score breakdown if available
    - "Edited" indicator when CV was modified
    - Retest button when edits detected
    """
    analysis = st.session_state.get("current_match_score") or result.get("match_analysis")
    if not analysis:
        return

    score = analysis.get("relevance_score", 0)
    is_edited = st.session_state.get("cv_was_edited", False)
    is_retesting = st.session_state.get("is_retesting", False)

    # Score color
    if score >= 0.7:
        score_color = "#28a745"  # Green
    elif score >= 0.5:
        score_color = "#ffc107"  # Yellow/Orange
    else:
        score_color = "#dc3545"  # Red

    # Build score card
    with st.container():
        # Use columns for layout
        col_score, col_status, col_action = st.columns([2, 2, 1])

        with col_score:
            # Score display with optional fade when edited
            opacity = "0.5" if is_edited else "1.0"
            st.markdown(
                f"""
                <div style="opacity: {opacity}; transition: opacity 0.3s;">
                    <span style="font-size: 0.9rem; color: #666;">ATS Match Score</span>
                    <div style="font-size: 2.5rem; font-weight: bold; color: {score_color};">
                        {score:.0%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_status:
            # Show breakdown or knockout status
            if analysis.get("knockout_triggered"):
                st.markdown(
                    f"""
                    <div style="padding: 0.5rem; background: #fff0f0; border-radius: 4px; opacity: {opacity};">
                        <span style="color: #dc3545;">**Knockout:** {analysis.get("knockout_reason", "Missing required skills")}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif analysis.get("score_breakdown"):
                breakdown = analysis["score_breakdown"]
                algo_score = analysis.get("algorithmic_score", 0)
                llm_adj = analysis.get("llm_adjustment", 0)
                adj_str = f"+{llm_adj:.0%}" if llm_adj > 0 else f"{llm_adj:.0%}"

                st.markdown(
                    f"""
                    <div style="font-size: 0.85rem; color: #666; opacity: {opacity};">
                        <div>Skills: {breakdown.get("exact_skill_match", 0):.0%} | Experience: {breakdown.get("experience_years_fit", 0):.0%}</div>
                        <div>Education: {breakdown.get("education_match", 0):.0%} | Recency: {breakdown.get("recency_score", 0):.0%}</div>
                        <div style="margin-top: 0.25rem; font-style: italic;">Algo: {algo_score:.0%}, LLM: {adj_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Show edited indicator
            if is_edited:
                st.markdown(
                    """
                    <div style="margin-top: 0.5rem; padding: 0.25rem 0.5rem; background: #fff3cd; border-radius: 4px; display: inline-block;">
                        <span style="color: #856404;">&#9998; Edited - score may not be accurate</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_action:
            # Retest button (only visible when edited)
            if is_edited and not is_retesting:
                if st.button(
                    "Retest", key="retest_score_btn", type="primary", use_container_width=True
                ):
                    st.session_state.is_retesting = True
                    st.rerun()

    # Handle retest if triggered
    if is_retesting:
        with st.spinner("Re-scoring edited CV..."):
            new_analysis = _run_retest_scoring(
                st.session_state.edited_cv,
                result,
            )
            if new_analysis:
                st.session_state.current_match_score = new_analysis
                st.session_state.cv_was_edited = False
                st.session_state.original_cv_content = st.session_state.edited_cv
            st.session_state.is_retesting = False
            st.rerun()

    st.divider()


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

    # Initialize edit tracking state
    if "cv_was_edited" not in st.session_state:
        st.session_state.cv_was_edited = False
    if "original_cv_content" not in st.session_state:
        st.session_state.original_cv_content = result.get("tailored_cv", "")
    if "current_match_score" not in st.session_state:
        st.session_state.current_match_score = result.get("match_analysis", {})
    if "is_retesting" not in st.session_state:
        st.session_state.is_retesting = False

    # Reset edit tracking when new CV is generated
    if st.session_state.get("last_cv_hash") != hash(result.get("tailored_cv", "")):
        st.session_state.cv_was_edited = False
        st.session_state.original_cv_content = result.get("tailored_cv", "")
        st.session_state.current_match_score = result.get("match_analysis", {})
        # Reset cover letter when new CV is generated
        st.session_state.cover_letter = None
        st.session_state.cover_letter_result = None

    # Initialize cover letter state
    if "cover_letter" not in st.session_state:
        st.session_state.cover_letter = None
    if "cover_letter_result" not in st.session_state:
        st.session_state.cover_letter_result = None
    if "is_generating_cover_letter" not in st.session_state:
        st.session_state.is_generating_cover_letter = False

    # Initialize active tab state for programmatic switching
    if "active_result_tab" not in st.session_state:
        st.session_state.active_result_tab = "Tailored CV"

    # Tab options
    tab_options = ["Tailored CV", "Cover Letter", "Match Analysis", "Tailoring Plan"]

    # Tab selector using segmented control (allows programmatic switching via session state)
    active_tab = st.radio(
        "View",
        tab_options,
        index=tab_options.index(st.session_state.active_result_tab),
        horizontal=True,
        key="result_tab_selector",
        label_visibility="collapsed",
    )

    # Update session state if user manually switched tabs
    if active_tab != st.session_state.active_result_tab:
        st.session_state.active_result_tab = active_tab

    # Render content based on active tab
    if active_tab == "Tailored CV":
        render_tailored_cv(result)
    elif active_tab == "Cover Letter":
        render_cover_letter(result)
    elif active_tab == "Match Analysis":
        render_match_analysis(result)
    elif active_tab == "Tailoring Plan":
        render_tailoring_plan(result)


def render_tailored_cv(result: dict[str, Any]) -> None:
    """Render the tailored CV tab with editing and export options."""
    if not result.get("tailored_cv"):
        st.warning("No tailored CV was generated.")
        return

    # Initialize session state for edited CV content
    if "edited_cv" not in st.session_state:
        st.session_state.edited_cv = result["tailored_cv"]

    # Reset edited content if result changed (new generation)
    if st.session_state.get("last_cv_hash") != hash(result["tailored_cv"]):
        st.session_state.edited_cv = result["tailored_cv"]
        st.session_state.last_cv_hash = hash(result["tailored_cv"])

    # --- Match Score Display with Edit Tracking ---
    _render_match_score_card(result)

    st.subheader("Your Tailored CV")

    # Toggle between edit and preview mode
    col_toggle1, _ = st.columns([1, 4])
    with col_toggle1:
        edit_mode = st.toggle("Edit mode", value=False, key="cv_edit_mode")

    if edit_mode:
        # Editable text area
        new_content = st.text_area(
            "Edit your CV (Markdown format)",
            value=st.session_state.edited_cv,
            height=500,
            key="cv_editor",
            help="Edit the markdown content. Changes are preserved until you generate a new CV.",
        )

        # Check if content was changed
        if new_content != st.session_state.original_cv_content:
            st.session_state.cv_was_edited = True
        st.session_state.edited_cv = new_content

        # Show preview in expander when editing
        with st.expander("Preview", expanded=False):
            st.markdown(st.session_state.edited_cv)
    else:
        # Preview mode - show rendered markdown
        st.markdown(st.session_state.edited_cv)

    st.divider()

    # Download buttons
    st.write("**Download Options**")
    col1, col2, col3 = st.columns(3)

    # Create Cover Letter button (full width, above download buttons)
    col_cl_btn, col_cl_status = st.columns([1, 2])
    with col_cl_btn:
        if st.button(
            "Create Cover Letter",
            key="create_cover_letter_btn",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("is_generating_cover_letter", False),
        ):
            # Store the edited CV to be used for cover letter generation
            st.session_state.cover_letter_source_cv = st.session_state.edited_cv
            st.session_state.is_generating_cover_letter = True
            # Switch to Cover Letter tab
            st.session_state.active_result_tab = "Cover Letter"
            st.rerun()
    with col_cl_status:
        if st.session_state.get("cover_letter"):
            st.success("Cover letter ready! See the **Cover Letter** tab.")

    st.write("")  # Small spacing

    # Get the content to export (edited version)
    cv_content = st.session_state.edited_cv

    with col1:
        st.download_button(
            label="Download Markdown",
            data=cv_content,
            file_name="tailored_cv.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col2:
        # Generate PDF on demand
        try:
            pdf_bytes = generate_cv_pdf(cv_content)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name="tailored_cv.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

    with col3:
        # Plain text option (strips markdown)
        plain_text = cv_content
        plain_text = re.sub(r"^#{1,6}\s+", "", plain_text, flags=re.MULTILINE)  # Headers
        plain_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", plain_text)  # Bold
        plain_text = re.sub(r"\*([^*]+)\*", r"\1", plain_text)  # Italic
        plain_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain_text)  # Links

        st.download_button(
            label="Download Plain Text",
            data=plain_text,
            file_name="tailored_cv.txt",
            mime="text/plain",
            use_container_width=True,
        )


def _generate_cover_letter(result: dict[str, Any], char_limit: int) -> CoverLetterGenerationResult:
    """Generate cover letter using the CoverLetterGenerator.

    Args:
        result: The workflow result containing tailored CV, job requirements, etc.
        char_limit: Target character limit for the cover letter.

    Returns:
        CoverLetterGenerationResult with reasoning and final output.
    """
    from cv_warlock.llm.base import get_llm_provider
    from cv_warlock.processors.cover_letter import CoverLetterGenerator

    # Get processing params from session state
    params = st.session_state.get("process_params", {})
    if not params:
        raise ValueError("Missing processing parameters. Please generate a CV first.")

    # Create LLM provider
    llm_provider = get_llm_provider(
        provider=params.get("provider", "anthropic"),
        model=params.get("model", "claude-sonnet-4-5-20250929"),
        api_key=params.get("api_key"),
    )

    # Create generator
    generator = CoverLetterGenerator(llm_provider)

    # Use edited CV if available (from Create Cover Letter button), otherwise use result
    tailored_cv = st.session_state.get("cover_letter_source_cv") or result.get("tailored_cv", "")

    # Generate cover letter
    return generator.generate(
        tailored_cv=tailored_cv,
        job_requirements=result.get("job_requirements"),
        match_analysis=result.get("match_analysis", {}),
        character_limit=char_limit,
    )


def render_cover_letter(result: dict[str, Any]) -> None:
    """Render the cover letter tab with generation and download options.

    Args:
        result: The workflow result containing tailored CV, job requirements, etc.
    """
    # Check if CV was generated
    if not result.get("tailored_cv"):
        st.info("Generate a tailored CV first, then create a cover letter.")
        return

    st.subheader("Cover Letter Generator")

    st.markdown(
        "Generate a **plain text** cover letter optimized for job application forms. "
        "Uses your tailored CV and the job requirements for context."
    )

    # Character limit from session state
    char_limit = st.session_state.get("cover_letter_char_limit", 2500)
    st.caption(f"Target length: **{char_limit:,}** characters (set in sidebar)")

    # Generate button
    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        generate_btn = st.button(
            "Generate Cover Letter",
            type="primary",
            disabled=st.session_state.is_generating_cover_letter,
            use_container_width=True,
        )

    # Phase 1: Button clicked - set flag and rerun
    if generate_btn and not st.session_state.is_generating_cover_letter:
        st.session_state.is_generating_cover_letter = True
        st.rerun()

    # Phase 2: Actually generate
    if st.session_state.is_generating_cover_letter:
        with st.spinner("Generating cover letter with CoT reasoning..."):
            try:
                cover_letter_result = _generate_cover_letter(result, char_limit)
                st.session_state.cover_letter = cover_letter_result.final_cover_letter
                st.session_state.cover_letter_result = cover_letter_result
            except Exception as e:
                st.error(f"Cover letter generation failed: {e}")
                st.session_state.cover_letter = None
                st.session_state.cover_letter_result = None
            finally:
                st.session_state.is_generating_cover_letter = False
                st.rerun()

    # Display cover letter if available
    if st.session_state.cover_letter:
        st.divider()

        final_text = st.session_state.cover_letter
        char_count = len(final_text)

        # Character count display
        col_count, col_status = st.columns([1, 3])
        with col_count:
            if char_count <= char_limit:
                delta_text = f"{char_limit - char_count:,} under limit"
                st.metric("Characters", f"{char_count:,}", delta=delta_text)
            else:
                delta_text = f"{char_count - char_limit:,} over limit"
                st.metric("Characters", f"{char_count:,}", delta=delta_text, delta_color="inverse")

        # Display in text area (allows easy copy)
        st.text_area(
            "Cover Letter (click to select all, then copy)",
            value=final_text,
            height=400,
            key="cover_letter_display",
            help="Plain text ready to paste into job applications",
        )

        # Download and action buttons
        col_download, col_regenerate = st.columns(2)

        with col_download:
            st.download_button(
                label="Download as .txt",
                data=final_text,
                file_name="cover_letter.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col_regenerate:
            if st.button("Regenerate", use_container_width=True):
                st.session_state.cover_letter = None
                st.session_state.cover_letter_result = None
                st.session_state.is_generating_cover_letter = True
                st.rerun()

        # Show reasoning in expander
        cover_letter_result = st.session_state.cover_letter_result
        if cover_letter_result and hasattr(cover_letter_result, "reasoning"):
            with st.expander("View Reasoning", expanded=False):
                reasoning = cover_letter_result.reasoning
                st.markdown(f"**Opening Hook:** {reasoning.opening_hook}")
                st.markdown("**Key Selling Points:**")
                for point in reasoning.key_selling_points:
                    st.markdown(f"- {point}")
                st.markdown(f"**Company Connection:** {reasoning.company_connection}")
                st.markdown(f"**Featured Metric:** {reasoning.metric_to_feature}")
                st.markdown(f"**Call to Action:** {reasoning.call_to_action}")
                st.markdown(f"**Tone Guidance:** {reasoning.tone_guidance}")


def render_match_analysis(result: dict[str, Any]) -> None:
    """Render the match analysis tab."""
    if not result.get("match_analysis"):
        st.warning("No match analysis available.")
        return

    analysis = result["match_analysis"]

    # Score display
    score = analysis["relevance_score"]

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
