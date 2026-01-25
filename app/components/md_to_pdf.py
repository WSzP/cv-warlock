"""Markdown to PDF conversion tool component."""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from app.utils.pdf_generator import CVStyle, generate_cv_pdf


def _extract_cv_filename(markdown: str, suffix: str = "cv") -> str:
    """Extract name from CV markdown and create a filename-safe string.

    Args:
        markdown: The CV markdown content.
        suffix: Suffix to append (e.g., 'cv', 'cover-letter').

    Returns:
        Filename like 'peter-w-szabo-cv' (without extension).
    """
    # Look for # Name as the first heading
    match = re.search(r"^#\s+(.+?)$", markdown, re.MULTILINE)
    if match:
        name = match.group(1).strip()
        # Convert to lowercase, replace spaces/special chars with hyphens
        filename = re.sub(r"[^\w\s-]", "", name.lower())
        filename = re.sub(r"[\s_]+", "-", filename)
        filename = re.sub(r"-+", "-", filename).strip("-")
        if filename:
            return f"{filename}-{suffix}"
    return suffix


# Project root for loading sample files
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Style display names and descriptions (Modern first = default)
STYLE_OPTIONS = {
    CVStyle.MODERN: {
        "label": "Modern",
        "description": "Contemporary design with accent colors and refined visual hierarchy",
    },
    CVStyle.PLAIN: {
        "label": "Plain",
        "description": "Classic, clean layout with centered header and subtle underlines",
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

    # Determine content based on checkbox state
    if use_sample:
        # Load fresh sample each time checkbox is checked
        default_content = _load_sample_cv()
    else:
        default_content = ""

    md_content = st.text_area(
        "Paste or edit your markdown CV:",
        value=default_content,
        height=400,
        placeholder="# Your Name\nEmail: your@email.com\n\n## Summary\n...",
        key=f"md_to_pdf_input_{use_sample}",  # Different key forces refresh
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
                pdf_bytes, page_count = generate_cv_pdf(md_content, style=selected_style)

                # Store in session state for download
                st.session_state["generated_pdf"] = pdf_bytes
                st.session_state["pdf_generated"] = True
                st.session_state["pdf_style"] = selected_style.value
                st.session_state["pdf_source_md"] = md_content  # Store for filename extraction
                st.session_state["pdf_page_count"] = page_count
                st.success(
                    f"PDF generated successfully with {STYLE_OPTIONS[selected_style]['label']} style!"
                )

            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                st.session_state["pdf_generated"] = False

    # Show download button if PDF was generated
    if st.session_state.get("pdf_generated") and st.session_state.get("generated_pdf"):
        with col2:
            # Generate filename from CV name
            source_md = st.session_state.get("pdf_source_md", "")
            base_filename = _extract_cv_filename(source_md, "cv")
            page_count = st.session_state.get("pdf_page_count", 1)
            page_label = "page" if page_count == 1 else "pages"
            st.download_button(
                label=f"Download PDF ({page_count} {page_label})",
                data=st.session_state["generated_pdf"],
                file_name=f"{base_filename}.pdf",
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
