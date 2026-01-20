"""Markdown to PDF conversion tool component."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.utils.pdf_generator import CVStyle, generate_cv_pdf

# Project root for loading sample files
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Style display names and descriptions
STYLE_OPTIONS = {
    CVStyle.PLAIN: {
        "label": "Plain",
        "description": "Classic, clean layout with centered header and subtle underlines",
    },
    CVStyle.MODERN: {
        "label": "Modern",
        "description": "Contemporary design with accent colors and refined visual hierarchy",
    },
}


def _load_sample_cv() -> str:
    """Load sample CV from examples directory."""
    sample_path = PROJECT_ROOT / "examples" / "sample_cv.md"
    if sample_path.exists():
        return sample_path.read_text(encoding="utf-8")
    return "# Sample CV not found\n\nPlease check examples/sample_cv.md exists."


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

    # Input area
    st.subheader("Input Markdown")

    use_sample = st.checkbox("Load sample markdown", value=False, key="md_to_pdf_sample")

    if use_sample:
        sample_md = _load_sample_cv()
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

    # Style selection
    st.subheader("PDF Style")

    style_col1, style_col2 = st.columns(2)

    with style_col1:
        selected_style = st.radio(
            "Choose a style:",
            options=list(STYLE_OPTIONS.keys()),
            format_func=lambda x: STYLE_OPTIONS[x]["label"],
            key="md_to_pdf_style",
            horizontal=True,
        )

    with style_col2:
        st.caption(STYLE_OPTIONS[selected_style]["description"])

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
                pdf_bytes = generate_cv_pdf(md_content, style=selected_style)

                # Store in session state for download
                st.session_state["generated_pdf"] = pdf_bytes
                st.session_state["pdf_generated"] = True
                st.session_state["pdf_style"] = selected_style.value
                st.success(
                    f"PDF generated successfully with {STYLE_OPTIONS[selected_style]['label']} style!"
                )

            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                st.session_state["pdf_generated"] = False

    # Show download button if PDF was generated
    if st.session_state.get("pdf_generated") and st.session_state.get("generated_pdf"):
        with col2:
            style_suffix = st.session_state.get("pdf_style", "plain")
            st.download_button(
                label="Download PDF",
                data=st.session_state["generated_pdf"],
                file_name=f"cv_{style_suffix}.pdf",
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
