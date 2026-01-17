"""CV input component for Streamlit UI."""

import streamlit as st

from utils.linkedin_fetcher import fetch_linkedin_profile


SAMPLE_LINKEDIN_URL = "https://www.linkedin.com/in/wszabopeter/"

SAMPLE_CV = """# John Doe
**Software Engineer**

Email: john.doe@email.com | Phone: (555) 123-4567
LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe

## Summary
Experienced software engineer with 5+ years of experience building web applications and APIs. Proficient in Python, JavaScript, and cloud technologies.

## Experience

### Senior Software Engineer
**Tech Corp** | Jan 2022 - Present
- Led development of microservices architecture serving 1M+ daily users
- Reduced API response time by 40% through optimization
- Mentored team of 3 junior developers

### Software Engineer
**StartupXYZ** | Jun 2019 - Dec 2021
- Built REST APIs using Python and FastAPI
- Implemented CI/CD pipelines with GitHub Actions
- Developed React frontend components

## Education
**B.S. Computer Science** - State University, 2019

## Skills
Python, JavaScript, TypeScript, React, Node.js, FastAPI, PostgreSQL, AWS, Docker, Git
"""


def on_sample_cv_change():
    """Handle sample CV checkbox change."""
    if st.session_state.use_sample_cv_checkbox:
        st.session_state.cv_text_area = SAMPLE_CV
    else:
        st.session_state.cv_text_area = ""


def on_sample_linkedin_change():
    """Handle sample LinkedIn URL checkbox change."""
    if st.session_state.use_sample_linkedin_checkbox:
        st.session_state.linkedin_url_input = SAMPLE_LINKEDIN_URL
    else:
        st.session_state.linkedin_url_input = ""


def render_cv_input() -> str:
    """Render the CV input component.

    Returns:
        str: The CV text entered by the user.
    """
    st.subheader("Your CV")

    # Initialize session state
    if "cv_text_area" not in st.session_state:
        st.session_state.cv_text_area = ""
    if "linkedin_url_input" not in st.session_state:
        st.session_state.linkedin_url_input = ""

    # Input method selection
    input_method = st.radio(
        "Input method",
        options=["Paste Text", "Import from LinkedIn"],
        horizontal=True,
        key="cv_input_method",
    )

    if input_method == "Import from LinkedIn":
        # LinkedIn import mode
        st.checkbox(
            "Use sample LinkedIn URL",
            key="use_sample_linkedin_checkbox",
            on_change=on_sample_linkedin_change,
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            linkedin_url = st.text_input(
                "LinkedIn Profile URL",
                placeholder="https://www.linkedin.com/in/username/",
                key="linkedin_url_input",
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Align button
            fetch_clicked = st.button(
                "Fetch CV",
                type="primary",
                disabled=not linkedin_url,
                use_container_width=True,
            )

        if fetch_clicked and linkedin_url:
            with st.spinner("Fetching LinkedIn profile..."):
                cv_text, error = fetch_linkedin_profile(linkedin_url)

                if error:
                    st.error(error)
                    st.info(
                        "**Tip:** LinkedIn may block automated access. You can:\n"
                        "1. Use LinkedIn's 'Save to PDF' feature\n"
                        "2. Copy your profile text manually\n"
                        "3. Switch to 'Paste Text' mode"
                    )
                elif cv_text:
                    st.session_state.cv_text_area = cv_text
                    st.success("Profile imported successfully!")
                    st.rerun()

        # Show editable text area with imported content
        if st.session_state.cv_text_area:
            st.markdown("**Imported CV** (edit below if needed):")
            cv_text = st.text_area(
                "CV content",
                value=st.session_state.cv_text_area,
                height=300,
                key="linkedin_cv_preview",
                label_visibility="collapsed",
            )
            # Sync back to main state
            st.session_state.cv_text_area = cv_text

            if cv_text:
                word_count = len(cv_text.split())
                st.caption(f"{word_count} words")

            return cv_text

        return st.session_state.cv_text_area

    else:
        # Paste text mode
        st.checkbox(
            "Use sample CV",
            key="use_sample_cv_checkbox",
            on_change=on_sample_cv_change,
        )

        cv_text = st.text_area(
            "Paste your CV here (Markdown or plain text)",
            placeholder="Paste your CV content here...",
            height=400,
            key="cv_text_area",
        )

        if cv_text:
            word_count = len(cv_text.split())
            st.caption(f"{word_count} words")

        return cv_text
