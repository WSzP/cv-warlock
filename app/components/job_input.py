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


def render_job_input() -> str:
    """Render the job specification input component.

    Returns:
        str: The job specification text entered by the user.
    """
    st.subheader("Job Specification")

    # Option to use sample
    use_sample = st.checkbox("Use sample job posting", key="use_sample_job")

    if use_sample:
        job_text = st.text_area(
            "Paste the job posting here",
            value=SAMPLE_JOB,
            height=400,
            key="job_input",
        )
    else:
        job_text = st.text_area(
            "Paste the job posting here",
            placeholder="Paste the job description here...",
            height=400,
            key="job_input",
        )

    if job_text:
        word_count = len(job_text.split())
        st.caption(f"{word_count} words")

    return job_text
