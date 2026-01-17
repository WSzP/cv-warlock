"""CV input component for Streamlit UI."""

import streamlit as st

from utils.pdf_parser import extract_text_from_pdf


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


def render_cv_input() -> str:
    """Render the CV input component.

    Returns:
        str: The CV text entered by the user.
    """
    st.subheader("Your CV")

    # Initialize session state
    if "cv_text_area" not in st.session_state:
        st.session_state.cv_text_area = ""

    # Input method selection
    input_method = st.radio(
        "Input method",
        options=["Paste Text", "Upload PDF"],
        horizontal=True,
        key="cv_input_method",
    )

    if input_method == "Upload PDF":
        # PDF upload mode
        st.info(
            "**From LinkedIn:** Go to your profile → Click 'More' → "
            "Select 'Save to PDF' → Upload here."
        )

        uploaded_file = st.file_uploader(
            "Upload your CV (PDF)",
            type=["pdf"],
            key="cv_pdf_upload",
        )

        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                cv_text, error = extract_text_from_pdf(uploaded_file)

                if error:
                    st.error(f"Error reading PDF: {error}")
                elif cv_text:
                    st.session_state.cv_text_area = cv_text
                    st.success("PDF content extracted!")

        # Show editable text area with extracted content
        if st.session_state.cv_text_area:
            st.markdown("**Extracted CV** (edit below if needed):")
            cv_text = st.text_area(
                "CV content",
                value=st.session_state.cv_text_area,
                height=300,
                key="pdf_cv_preview",
                label_visibility="collapsed",
            )
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
