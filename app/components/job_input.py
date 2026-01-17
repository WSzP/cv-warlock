"""Job specification input component for Streamlit UI."""

import streamlit as st


SAMPLE_JOB = """# Senior Python Developer

**Company:** Acme Tech
**Location:** Remote
**Type:** Full-time

## About the Role
We're looking for a Senior Python Developer to join our growing engineering team. You'll work on building scalable backend services and APIs.

## Requirements

### Required
- 5+ years of experience with Python
- Experience with FastAPI or Django
- Strong knowledge of PostgreSQL or similar databases
- Experience with cloud platforms (AWS, GCP, or Azure)
- Experience with Docker and containerization
- Strong communication skills

### Nice to Have
- Experience with LangChain or other AI/ML frameworks
- Knowledge of TypeScript
- Experience with Kubernetes
- Contributions to open source projects

## Responsibilities
- Design and implement scalable backend services
- Write clean, maintainable code with proper testing
- Participate in code reviews and technical discussions
- Mentor junior team members
- Collaborate with product and design teams

## Benefits
- Competitive salary
- Remote-first culture
- Health insurance
- Learning budget
"""


def on_sample_job_change():
    """Handle sample job checkbox change."""
    if st.session_state.use_sample_job_checkbox:
        st.session_state.job_text_area = SAMPLE_JOB
    else:
        st.session_state.job_text_area = ""


def render_job_input() -> str:
    """Render the job specification input component.

    Returns:
        str: The job specification text entered by the user.
    """
    st.subheader("Job Specification")

    # Initialize text area state if needed
    if "job_text_area" not in st.session_state:
        st.session_state.job_text_area = ""

    # Initialize assume all tech skills checkbox (checked by default)
    if "assume_all_tech_skills" not in st.session_state:
        st.session_state.assume_all_tech_skills = True

    # Checkbox with callback
    st.checkbox(
        "Use sample job posting",
        key="use_sample_job_checkbox",
        on_change=on_sample_job_change,
    )

    # Text area
    job_text = st.text_area(
        "Paste the job posting here",
        placeholder="Paste the job description here...",
        height=400,
        key="job_text_area",
    )

    if job_text:
        word_count = len(job_text.split())
        st.caption(f"{word_count} words")

    # Assume all tech skills checkbox
    st.checkbox(
        "Assume all requested tech skills",
        key="assume_all_tech_skills",
        help=(
            "When enabled, assumes you have ALL technical skills listed in the job posting. "
            "Useful when your CV doesn't list every library/framework you know (there are thousands!), "
            "or when you haven't imported skills from LinkedIn. The tailored CV will include all "
            "required tech skills as if you have them."
        ),
    )

    return job_text
