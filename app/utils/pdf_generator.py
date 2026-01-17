"""PDF generator for tailored CVs with AI-friendly structure.

Creates well-structured PDFs optimized for:
- ATS (Applicant Tracking Systems) parsing
- AI analysis and extraction
- Human readability

Key structural elements:
- Clear section hierarchy (Name → Contact → Summary → Experience → Skills → Education)
- Consistent date/location formatting
- Semantic bullet points
- Clean, readable fonts (Unicode TTF for full character support)
- Proper PDF metadata
"""

import re
from pathlib import Path
from typing import Any

from fpdf import FPDF  # type: ignore[import-untyped]

# Get the fonts directory (relative to project root)
_FONTS_DIR = Path(__file__).parent.parent.parent / "fonts"


def _get_poppins_fonts() -> dict[str, Path] | None:
    """Get Poppins font files from the local fonts directory.

    Returns dict with font style keys ('regular', 'bold', 'italic', 'bold_italic')
    mapped to their file paths, or None if fonts not found.
    """
    if not _FONTS_DIR.exists():
        return None

    fonts = {
        "regular": _FONTS_DIR / "Poppins-Regular.ttf",
        "bold": _FONTS_DIR / "Poppins-Bold.ttf",
        "italic": _FONTS_DIR / "Poppins-Italic.ttf",
        "bold_italic": _FONTS_DIR / "Poppins-BoldItalic.ttf",
    }

    # Verify all fonts exist
    for path in fonts.values():
        if not path.exists():
            return None

    return fonts


class CVPDFGenerator(FPDF):
    """PDF generator optimized for CV structure and AI parsing.

    Uses Poppins font (Open Font License) for full international character
    support including Romanian diacritics (ț, ș, ă, â, î), accented characters,
    and other special symbols.
    """

    def __init__(self) -> None:
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(left=20, top=20, right=20)

        # Load Poppins font for professional, modern look
        self._setup_poppins_font()

    def _setup_poppins_font(self) -> None:
        """Set up Poppins font for the PDF."""
        fonts = _get_poppins_fonts()

        if fonts:
            # Add Poppins with proper bold/italic variants
            self.add_font("Poppins", "", str(fonts["regular"]))
            self.add_font("Poppins", "B", str(fonts["bold"]))
            self.add_font("Poppins", "I", str(fonts["italic"]))
            self.add_font("Poppins", "BI", str(fonts["bold_italic"]))
            self.font_name = "Poppins"
        else:
            # Fallback to built-in Helvetica (limited character support)
            self.font_name = "Helvetica"

    def header(self) -> None:
        """Minimal header - CVs shouldn't have page headers."""
        pass

    def footer(self) -> None:
        """Add page number footer for multi-page CVs."""
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font(self.font_name, "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")
            self.set_text_color(0, 0, 0)

    def add_name(self, name: str) -> None:
        """Add candidate name as main heading (H1 equivalent)."""
        self.set_font(self.font_name, "B", 18)
        self.cell(0, 12, name.strip(), ln=True, align="C")
        self.ln(2)

    def add_contact_line(self, contact: str) -> None:
        """Add contact info line (centered, smaller font)."""
        self.set_font(self.font_name, "", 10)
        self.set_text_color(64, 64, 64)
        self.cell(0, 6, contact.strip(), ln=True, align="C")
        self.set_text_color(0, 0, 0)

    def add_section_header(self, title: str) -> None:
        """Add section header (H2 equivalent) with underline."""
        self.ln(6)
        self.set_font(self.font_name, "B", 12)
        self.cell(0, 8, title.upper(), ln=True)
        # Add subtle underline
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def add_experience_header(self, title: str, company: str, date_location: str) -> None:
        """Add experience entry header with title, company, and date/location."""
        self.set_font(self.font_name, "B", 11)
        self.cell(0, 6, title.strip(), ln=True)

        self.set_font(self.font_name, "", 10)
        # Company and date on same line
        company_clean = company.strip()
        company_width = self.get_string_width(company_clean) + 2
        self.cell(company_width, 5, company_clean)
        self.set_text_color(96, 96, 96)
        self.cell(0, 5, date_location.strip(), ln=True, align="R")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def add_bullet_point(self, text: str, indent: int = 0) -> None:
        """Add a bullet point with proper formatting."""
        self.set_font(self.font_name, "", 10)
        self.set_x(self.l_margin)  # Reset to left margin

        bullet_indent = 5 + (indent * 5)  # Indent for nested bullets

        # Write bullet with indent
        self.cell(bullet_indent, 5, "")  # Indent space
        self.cell(5, 5, "•")  # Bullet character

        # Calculate available width for text (from current position to right margin)
        text_start_x = self.get_x()
        available_width = self.w - self.r_margin - text_start_x

        # Text with word wrap
        self.multi_cell(available_width, 5, text.strip())

    def add_paragraph(self, text: str) -> None:
        """Add a regular paragraph."""
        self.set_font(self.font_name, "", 10)
        self.set_x(self.l_margin)  # Reset to left margin
        self.multi_cell(0, 5, text.strip())
        self.ln(2)

    def add_skill_line(self, category: str, skills: str) -> None:
        """Add a skill category line (e.g., 'Languages: Python, TypeScript')."""
        self.set_font(self.font_name, "B", 10)
        self.set_x(self.l_margin)  # Reset to left margin
        cat_width = self.get_string_width(category + ": ") + 2
        self.cell(cat_width, 5, f"{category}: ")
        self.set_font(self.font_name, "", 10)
        # Calculate remaining width
        available_width = self.w - self.r_margin - self.get_x()
        self.multi_cell(available_width, 5, skills.strip())


def parse_markdown_cv(markdown: str) -> dict[str, Any]:
    """Parse markdown CV into structured sections.

    Returns a dict with:
    - name: Candidate name (from H1)
    - contact: Contact info lines
    - sections: List of {header, content} dicts
    """
    lines = markdown.strip().split("\n")
    result: dict[str, Any] = {
        "name": "",
        "contact": [],
        "sections": [],
    }

    current_section: dict[str, Any] | None = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # H1: Name (# Name or Name\n===)
        if line.startswith("# ") and not result["name"]:
            result["name"] = line[2:].strip()
            i += 1
            continue

        # Check for underline-style H1
        if i + 1 < len(lines) and lines[i + 1].strip().startswith("==="):
            result["name"] = line
            i += 2
            continue

        # H2: Section header (## Section or Section\n---)
        if line.startswith("## "):
            if current_section:
                result["sections"].append(current_section)
            current_section = {"header": line[3:].strip(), "content": []}
            i += 1
            continue

        # Check for underline-style H2
        if i + 1 < len(lines) and lines[i + 1].strip().startswith("---") and line:
            if current_section:
                result["sections"].append(current_section)
            current_section = {"header": line, "content": []}
            i += 2
            continue

        # Contact info: lines before first section, after name
        if result["name"] and not current_section and line:
            # Skip horizontal rules
            if not re.match(r"^[-=_*]{3,}$", line):
                result["contact"].append(line)
            i += 1
            continue

        # Content lines for current section
        if current_section is not None and line:
            current_section["content"].append(lines[i])  # Keep original indentation
        elif current_section is not None and not line:
            current_section["content"].append("")  # Preserve blank lines

        i += 1

    # Don't forget the last section
    if current_section:
        result["sections"].append(current_section)

    return result


def generate_cv_pdf(markdown: str) -> bytes:
    """Generate a well-structured PDF from markdown CV.

    Args:
        markdown: The CV content in markdown format.

    Returns:
        PDF content as bytes.
    """
    parsed = parse_markdown_cv(markdown)
    pdf = CVPDFGenerator()

    # Set PDF metadata for better indexing
    pdf.set_title(f"CV - {parsed['name']}" if parsed["name"] else "Tailored CV")
    pdf.set_author(parsed["name"] or "")
    pdf.set_subject("Curriculum Vitae")
    pdf.set_keywords("CV, Resume, Professional Experience")
    pdf.set_creator("CV Warlock")

    pdf.add_page()

    # Name
    if parsed["name"]:
        pdf.add_name(parsed["name"])

    # Contact info
    for contact in parsed["contact"]:
        # Clean markdown formatting from contact line
        clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", contact)  # Remove bold
        clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)  # Links to text
        clean = re.sub(r"[|•·]", " | ", clean)  # Normalize separators
        clean = re.sub(r"\s+", " ", clean).strip()
        if clean:
            pdf.add_contact_line(clean)

    # Sections
    for section in parsed["sections"]:
        pdf.add_section_header(section["header"])
        _render_section_content(pdf, section["header"], section["content"])

    # Return PDF as bytes
    return bytes(pdf.output())


def _render_section_content(pdf: CVPDFGenerator, header: str, content: list[str]) -> None:
    """Render section content with appropriate formatting based on section type."""
    header_lower = header.lower()

    # Experience/Work sections: parse job entries
    if any(kw in header_lower for kw in ["experience", "work", "employment", "history"]):
        _render_experience_section(pdf, content)
    # Skills sections: render as category lists
    elif any(kw in header_lower for kw in ["skill", "technical", "competenc"]):
        _render_skills_section(pdf, content)
    # Education: similar to experience but simpler
    elif any(kw in header_lower for kw in ["education", "academic", "qualification"]):
        _render_education_section(pdf, content)
    # Other sections: render as paragraphs/bullets
    else:
        _render_generic_section(pdf, content)


def _render_experience_section(pdf: CVPDFGenerator, content: list[str]) -> None:
    """Render experience section with job entries."""
    i = 0
    while i < len(content):
        line = content[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # H3 or bold line: likely job title
        if line.startswith("### ") or (line.startswith("**") and line.endswith("**")):
            title = re.sub(r"^###\s*", "", line)
            title = re.sub(r"^\*\*|\*\*$", "", title)

            company = ""
            date_location = ""

            # Next line might be company/date
            if i + 1 < len(content):
                next_line = content[i + 1].strip()
                # Check if it's italic (company) or contains date patterns
                if next_line.startswith("*") or re.search(r"\d{4}", next_line):
                    # Parse company | location | date pattern
                    parts = re.split(r"\s*[|•·]\s*", next_line.strip("*_ "))
                    if len(parts) >= 1:
                        company = parts[0]
                    if len(parts) >= 2:
                        date_location = " | ".join(parts[1:])
                    i += 1

            pdf.add_experience_header(title, company, date_location)
            i += 1
            continue

        # Bullet points
        if line.startswith(("-", "*", "•")) and not line.startswith("**"):
            bullet_text = re.sub(r"^[-*•]\s*", "", line)
            # Clean markdown formatting
            bullet_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", bullet_text)
            bullet_text = re.sub(r"\*([^*]+)\*", r"\1", bullet_text)
            pdf.add_bullet_point(bullet_text)
            i += 1
            continue

        # Regular text
        if line and not line.startswith("#"):
            clean_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            clean_text = re.sub(r"\*([^*]+)\*", r"\1", clean_text)
            pdf.add_paragraph(clean_text)

        i += 1


def _render_skills_section(pdf: CVPDFGenerator, content: list[str]) -> None:
    """Render skills section."""
    for line in content:
        line = line.strip()
        if not line:
            continue

        # Check for category: skills pattern
        if ":" in line and not line.startswith(("-", "*", "•")):
            parts = line.split(":", 1)
            category = parts[0].strip("*_ ")
            skills = parts[1].strip() if len(parts) > 1 else ""
            pdf.add_skill_line(category, skills)
        # Bullet point
        elif line.startswith(("-", "*", "•")):
            skill_text = re.sub(r"^[-*•]\s*", "", line)
            skill_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", skill_text)
            pdf.add_bullet_point(skill_text)
        # Regular text
        elif line:
            pdf.add_paragraph(line.strip("*_ "))


def _render_education_section(pdf: CVPDFGenerator, content: list[str]) -> None:
    """Render education section."""
    i = 0
    while i < len(content):
        line = content[i].strip()

        if not line:
            i += 1
            continue

        # H3 or bold: degree/institution
        if line.startswith("### ") or (line.startswith("**") and line.endswith("**")):
            title = re.sub(r"^###\s*", "", line)
            title = re.sub(r"^\*\*|\*\*$", "", title)

            institution = ""
            date_location = ""

            if i + 1 < len(content):
                next_line = content[i + 1].strip()
                if next_line and not next_line.startswith(("-", "*", "#")):
                    parts = re.split(r"\s*[|•·]\s*", next_line.strip("*_ "))
                    if len(parts) >= 1:
                        institution = parts[0]
                    if len(parts) >= 2:
                        date_location = " | ".join(parts[1:])
                    i += 1

            pdf.add_experience_header(title, institution, date_location)
            i += 1
            continue

        # Bullet points
        if line.startswith(("-", "*", "•")):
            text = re.sub(r"^[-*•]\s*", "", line)
            text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
            pdf.add_bullet_point(text)
            i += 1
            continue

        # Regular line (might be institution/date)
        if line and not line.startswith("#"):
            clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
            pdf.add_paragraph(clean)

        i += 1


def _render_generic_section(pdf: CVPDFGenerator, content: list[str]) -> None:
    """Render generic section (summary, certifications, etc.)."""
    for line in content:
        line = line.strip()
        if not line:
            continue

        # Bullet points
        if line.startswith(("-", "*", "•")) and not line.startswith("**"):
            text = re.sub(r"^[-*•]\s*", "", line)
            text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
            text = re.sub(r"\*([^*]+)\*", r"\1", text)
            pdf.add_bullet_point(text)
        # Regular text
        elif not line.startswith("#"):
            clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
            pdf.add_paragraph(clean)
