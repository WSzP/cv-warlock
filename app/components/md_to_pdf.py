"""Markdown to PDF conversion tool component."""

from __future__ import annotations

import streamlit as st

from app.utils.pdf_generator import generate_cv_pdf


def render_md_to_pdf_tool() -> None:
    """Render the markdown to PDF conversion tool.

    This tool allows users to convert markdown CVs to PDF format
    without going through the full tailoring process.
    """
    st.title("Markdown to PDF Converter")
    st.markdown(
        "Convert your markdown CV directly to a professionally formatted PDF. "
        "This uses the same PDF generation as the CV tailoring tool."
    )

    # Sample markdown for reference
    sample_md = """# John Doe
Email: john.doe@email.com | Phone: (555) 123-4567 | LinkedIn: linkedin.com/in/johndoe

## Summary
Experienced software engineer with 8+ years building scalable web applications...

## Experience

### Senior Software Engineer
**Tech Corp** | January 2022 - Present
- Led development of microservices architecture serving 1M+ users
- Reduced API response time by 40% through optimization

### Software Engineer
**StartupXYZ** | March 2019 - December 2021
- Built full-stack features using React and Python
- Mentored junior developers

## Skills
**Languages:** Python, TypeScript, Go
**Frameworks:** React, FastAPI, Django
**Tools:** Docker, Kubernetes, AWS

## Education

### Master of Science in Computer Science
**University of Technology** | 2017 - 2019
"""

    # Input area
    st.subheader("Input Markdown")

    use_sample = st.checkbox("Load sample markdown", value=False, key="md_to_pdf_sample")

    if use_sample:
        md_content = st.text_area(
            "Paste or edit your markdown CV:",
            value=sample_md,
            height=400,
            key="md_to_pdf_input",
        )
    else:
        md_content = st.text_area(
            "Paste your markdown CV:",
            value="",
            height=400,
            placeholder="# Your Name\nEmail: your@email.com\n\n## Summary\n...",
            key="md_to_pdf_input",
        )

    # Word count
    word_count = len(md_content.split()) if md_content.strip() else 0
    st.caption(f"Word count: {word_count}")

    st.divider()

    # Generate button and download
    col1, col2 = st.columns([1, 2])

    with col1:
        generate_clicked = st.button(
            "Generate PDF",
            type="primary",
            disabled=not md_content.strip(),
            use_container_width=True,
        )

    if generate_clicked and md_content.strip():
        with st.spinner("Generating PDF..."):
            try:
                pdf_bytes = generate_cv_pdf(md_content)

                # Store in session state for download
                st.session_state["generated_pdf"] = pdf_bytes
                st.session_state["pdf_generated"] = True
                st.success("PDF generated successfully!")

            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                st.session_state["pdf_generated"] = False

    # Show download button if PDF was generated
    if st.session_state.get("pdf_generated") and st.session_state.get("generated_pdf"):
        with col2:
            st.download_button(
                label="Download PDF",
                data=st.session_state["generated_pdf"],
                file_name="cv.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    # Tips section
    with st.expander("Markdown formatting tips"):
        st.markdown("""
**Supported format:**
- `# Name` - Your name as main heading
- Contact info as plain text after name
- `## Section` - Section headers (Experience, Skills, etc.)
- `### Job Title` - Job titles in experience section
- `**Company**` - Company names in bold
- `- Bullet points` - For achievements and descriptions

**Expected structure:**
```markdown
# Your Name
Email | Phone | LinkedIn

## Summary
Your professional summary...

## Experience
### Job Title
**Company** | Dates
- Achievement 1
- Achievement 2

## Skills
**Category:** Skill1, Skill2, Skill3

## Education
### Degree
**Institution** | Dates
```
        """)
