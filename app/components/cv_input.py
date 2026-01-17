"""CV input component for Streamlit UI."""

import streamlit as st


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


def render_cv_input() -> str:
    """Render the CV input component.

    Returns:
        str: The CV text entered by the user.
    """
    st.subheader("Your CV")

    # Option to use sample
    use_sample = st.checkbox("Use sample CV", key="use_sample_cv")

    if use_sample:
        cv_text = st.text_area(
            "Paste your CV here (Markdown or plain text)",
            value=SAMPLE_CV,
            height=400,
            key="cv_input",
        )
    else:
        cv_text = st.text_area(
            "Paste your CV here (Markdown or plain text)",
            placeholder="Paste your CV content here...",
            height=400,
            key="cv_input",
        )

    if cv_text:
        word_count = len(cv_text.split())
        st.caption(f"{word_count} words")

    return cv_text
