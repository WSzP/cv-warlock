"""PDF generator for tailored CVs with AI-friendly structure.

Creates well-structured PDFs optimized for:
- ATS (Applicant Tracking Systems) parsing
- AI analysis and extraction
- Human readability

Key structural elements:
- Clear section hierarchy (Name -> Contact -> Summary -> Experience -> Skills -> Education)
- Consistent date/location formatting
- Semantic bullet points
- Clean, readable fonts (Unicode TTF for full character support)
- Proper PDF metadata

Supported styles:
- plain: Classic, clean CV layout (original style)
- modern: Contemporary design with accent colors and refined visual hierarchy
"""

import re

from fpdf import ViewerPreferences  # type: ignore[import-untyped]

from app.utils.pdf_generator.generator import CVPDFGenerator
from app.utils.pdf_generator.markdown_parser import (
    _sanitize_markdown_bold,
    _sanitize_unsupported_chars,
    parse_markdown_cv,
)
from app.utils.pdf_generator.renderers import _render_section_content
from app.utils.pdf_generator.styles import STYLE_CONFIGS, CVStyle, StyleConfig

__all__ = [
    "CVStyle",
    "StyleConfig",
    "STYLE_CONFIGS",
    "CVPDFGenerator",
    "generate_cv_pdf",
    "parse_markdown_cv",
    "_sanitize_markdown_bold",
]


def generate_cv_pdf(markdown: str, style: CVStyle | str = CVStyle.MODERN) -> tuple[bytes, int]:
    """Generate a well-structured PDF from markdown CV.

    Args:
        markdown: The CV content in markdown format.
        style: The visual style to use ('plain' or 'modern'). Default is 'modern'.

    Returns:
        Tuple of (PDF content as bytes, page count).
    """
    # Convert string to CVStyle enum if needed
    if isinstance(style, str):
        style = CVStyle(style.lower())

    # Sanitize malformed markdown and unsupported characters before parsing
    markdown = _sanitize_markdown_bold(markdown)
    markdown = _sanitize_unsupported_chars(markdown)
    parsed = parse_markdown_cv(markdown)
    pdf = CVPDFGenerator(style=style)

    # Set PDF metadata for better indexing
    pdf.set_title(f"{parsed['name']} | CV" if parsed["name"] else "Tailored CV")
    pdf.set_author(parsed["name"] or "")
    pdf.set_subject("Curriculum Vitae")
    pdf.set_keywords("CV, Resume, Professional Experience")
    pdf.set_creator("CV Warlock")

    # Display document title in viewer (not filename)
    pdf.viewer_preferences = ViewerPreferences(display_doc_title=True)

    pdf.add_page()

    # Name
    if parsed["name"]:
        pdf.add_name(parsed["name"])

    # Contact info
    config = STYLE_CONFIGS[style]
    for contact in parsed["contact"]:
        # Clean markdown formatting from contact line
        clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", contact)  # Remove bold

        # Extract links before converting to plain text: [(display, url), ...]
        links: list[tuple[str, str]] = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", clean)

        # Convert markdown links to just display text
        clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)

        # Use style-appropriate separator
        clean = re.sub(r"[|•·]", config.contact_separator, clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        if clean:
            pdf.add_contact_line(clean, links if links else None)

    # Sections
    for section in parsed["sections"]:
        pdf.add_section_header(section["header"])
        _render_section_content(pdf, section["header"], section["content"])

    # Return PDF as bytes with page count
    return bytes(pdf.output()), pdf.page
