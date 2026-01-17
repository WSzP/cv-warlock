"""CV input component for Streamlit UI."""

from pathlib import Path

import streamlit as st

from utils.pdf_parser import extract_text_from_pdf


# Directory for saved CVs
CV_SAVE_DIR = Path(__file__).parent.parent / "data" / "cvs"


def _ensure_cv_dir():
    """Ensure the CV save directory exists."""
    CV_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _get_saved_cvs() -> list[str]:
    """Get list of saved CV files."""
    _ensure_cv_dir()
    return sorted([f.stem for f in CV_SAVE_DIR.glob("*.md")])


def _save_cv(name: str, content: str) -> bool:
    """Save CV to file."""
    _ensure_cv_dir()
    try:
        filepath = CV_SAVE_DIR / f"{name}.md"
        filepath.write_text(content, encoding="utf-8")
        return True
    except Exception:
        return False


def _load_cv(name: str) -> str | None:
    """Load CV from file."""
    try:
        filepath = CV_SAVE_DIR / f"{name}.md"
        if filepath.exists():
            return filepath.read_text(encoding="utf-8")
    except Exception:
        pass
    return None


def _delete_cv(name: str) -> bool:
    """Delete a saved CV."""
    try:
        filepath = CV_SAVE_DIR / f"{name}.md"
        if filepath.exists():
            filepath.unlink()
            return True
    except Exception:
        pass
    return False


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
    if "cv_name" not in st.session_state:
        st.session_state.cv_name = "my_cv"

    # Input method selection
    input_method = st.radio(
        "Input method",
        options=["Load Saved", "Paste Text", "Upload PDF"],
        horizontal=True,
        key="cv_input_method",
    )

    if input_method == "Load Saved":
        # Load saved CV mode
        saved_cvs = _get_saved_cvs()

        if not saved_cvs:
            st.info("No saved CVs found. Use 'Paste Text' or 'Upload PDF' to create one, then save it.")
            return ""

        # CV selection
        selected_cv = st.selectbox(
            "Select a saved CV",
            options=saved_cvs,
            key="selected_saved_cv",
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Load", type="primary", use_container_width=True):
                content = _load_cv(selected_cv)
                if content:
                    st.session_state.cv_text_area = content
                    st.session_state.cv_name = selected_cv
                    st.success(f"Loaded '{selected_cv}'")
                    st.rerun()
        with col2:
            if st.button("Delete", type="secondary", use_container_width=True):
                if _delete_cv(selected_cv):
                    st.success(f"Deleted '{selected_cv}'")
                    st.rerun()

        # Show loaded CV content
        if st.session_state.cv_text_area:
            st.markdown("---")
            st.markdown(f"**Editing: {st.session_state.cv_name}**")
            cv_text = st.text_area(
                "CV content",
                value=st.session_state.cv_text_area,
                height=350,
                key="loaded_cv_editor",
                label_visibility="collapsed",
            )
            st.session_state.cv_text_area = cv_text

            if cv_text:
                word_count = len(cv_text.split())
                st.caption(f"{word_count} words")

                # Save button
                if st.button("💾 Save Changes", use_container_width=True):
                    if _save_cv(st.session_state.cv_name, cv_text):
                        st.success(f"Saved '{st.session_state.cv_name}'")
                    else:
                        st.error("Failed to save CV")

            return cv_text

        return ""

    elif input_method == "Upload PDF":
        # PDF upload mode
        st.info(
            "**From LinkedIn:** Go to your profile → Click 'More' → "
            "Select 'Save to PDF' → Upload here."
        )

        # Romanian diacritics fix option
        fix_romanian = st.checkbox(
            "Fix Romanian diacritics from LinkedIn (ş→ș, ţ→ț)",
            value=True,
            key="fix_romanian_diacritics",
            help="LinkedIn uses incorrect cedilla characters (ş, ţ) instead of proper Romanian comma-below (ș, ț)",
        )

        uploaded_file = st.file_uploader(
            "Upload your CV (PDF)",
            type=["pdf"],
            key="cv_pdf_upload",
        )

        if uploaded_file is not None:
            # Track which file we've processed to avoid re-processing on every rerun
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("last_pdf_id") != file_id:
                with st.spinner("Extracting text from PDF..."):
                    cv_text, error = extract_text_from_pdf(uploaded_file, fix_romanian=fix_romanian)

                    if error:
                        st.error(f"Error reading PDF: {error}")
                    elif cv_text:
                        st.session_state.cv_text_area = cv_text
                        st.session_state.last_pdf_id = file_id
                        # Clear the text area widget state to force refresh
                        if "pdf_cv_preview" in st.session_state:
                            del st.session_state["pdf_cv_preview"]
                        st.success("PDF content extracted!")
                        st.rerun()

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

                # Save section
                st.markdown("---")
                col1, col2 = st.columns([3, 1])
                with col1:
                    save_name = st.text_input(
                        "Save as",
                        value=st.session_state.cv_name,
                        key="pdf_save_name",
                        label_visibility="collapsed",
                        placeholder="CV name (e.g., my_cv)",
                    )
                with col2:
                    if st.button("💾 Save", use_container_width=True):
                        if save_name:
                            if _save_cv(save_name, cv_text):
                                st.session_state.cv_name = save_name
                                st.success(f"Saved as '{save_name}'")
                            else:
                                st.error("Failed to save")
                        else:
                            st.error("Enter a name")

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
            height=350,
            key="cv_text_area",
        )

        if cv_text:
            word_count = len(cv_text.split())
            st.caption(f"{word_count} words")

            # Save section
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col1:
                save_name = st.text_input(
                    "Save as",
                    value=st.session_state.cv_name,
                    key="paste_save_name",
                    label_visibility="collapsed",
                    placeholder="CV name (e.g., my_cv)",
                )
            with col2:
                if st.button("💾 Save", use_container_width=True):
                    if save_name:
                        if _save_cv(save_name, cv_text):
                            st.session_state.cv_name = save_name
                            st.success(f"Saved as '{save_name}'")
                        else:
                            st.error("Failed to save")
                    else:
                        st.error("Enter a name")

        return cv_text
