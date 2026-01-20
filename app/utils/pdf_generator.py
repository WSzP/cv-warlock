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


def _sanitize_markdown_bold(text: str) -> str:
    """Fix malformed markdown bold markers.

    LLMs sometimes generate malformed bold like `*text:**` instead of `**text:**`.
    This function normalizes these patterns.
    """
    # Fix *text:** pattern (single asterisk start, double end)
    text = re.sub(r"(?<!\*)\*([^*]+):\*\*", r"**\1:**", text)
    # Fix **text:* pattern (double asterisk start, single end)
    text = re.sub(r"\*\*([^*]+):\*(?!\*)", r"**\1:**", text)
    # Fix orphaned asterisks around category names (e.g., "*Category:**")
    text = re.sub(r"(?<!\*)\*([A-Za-z][^*:]+):\*\*", r"**\1:**", text)
    return text


def _sanitize_unsupported_chars(text: str) -> str:
    """Replace characters not supported by Poppins font with ASCII equivalents.

    Poppins doesn't include certain Unicode symbols like arrows.
    """
    replacements = {
        "\u2192": "->",  # → Right arrow
        "\u2190": "<-",  # ← Left arrow
        "\u2194": "<->",  # ↔ Left-right arrow
        "\u21d2": "=>",  # ⇒ Double right arrow
        "\u21d0": "<=",  # ⇐ Double left arrow
        "\u2265": ">=",  # ≥ Greater than or equal
        "\u2264": "<=",  # ≤ Less than or equal
        "\u2260": "!=",  # ≠ Not equal
        "\u2026": "...",  # … Ellipsis
        "\u2014": "-",  # — Em dash
        "\u2013": "-",  # – En dash
        "\u2018": "'",  # ' Left single quote
        "\u2019": "'",  # ' Right single quote
        "\u201c": '"',  # " Left double quote
        "\u201d": '"',  # " Right double quote
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


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
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(left=20, top=20, right=20)

        # Load Poppins font for professional, modern look
        self._setup_poppins_font()

    def _safe_multi_cell(self, w: float, h: float, text: str, **kwargs: object) -> None:
        """Multi-cell with width validation to prevent fpdf errors."""
        # Width of 0 means "use remaining page width" which is always safe
        if w == 0:
            self.multi_cell(w, h, text, **kwargs)  # type: ignore[arg-type]
            return

        # Check if content can actually fit in the requested width
        # 1. Check strict minimum (40mm - reasonable column width)
        # 2. Check if any single word is wider than the column
        words = text.split()
        max_word_width = 0.0
        if words:
            try:
                max_word_width = max(self.get_string_width(word) for word in words)
            except Exception:
                # Fallback if get_string_width fails (e.g. encoding issues)
                max_word_width = 0.0

        is_too_narrow = w < 40
        has_wide_word = max_word_width > w

        if is_too_narrow or has_wide_word:
            # Column too narrow or word too wide - force new line and use full width
            self.ln()
            self.set_x(self.l_margin)
            # Use remaining page width (which is now full width sans margins)
            self.multi_cell(0, h, text, **kwargs)  # type: ignore[arg-type]
            return

        # Ensure minimum width of 20 units (absolute safety fallback)
        safe_width = max(w, 20)

        # If we're too close to right margin, start a new line
        if self.get_x() + safe_width > self.w - self.r_margin:
            self.ln()
            self.set_x(self.l_margin)
            # Recalculate width for new line (full width)
            # We call multi_cell(0) here to let FPDF handle the width
            self.multi_cell(0, h, text, **kwargs)  # type: ignore[arg-type]
            return

        self.multi_cell(safe_width, h, text, **kwargs)  # type: ignore[arg-type]

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
        self.set_x(self.l_margin)  # Reset to left margin
        self.multi_cell(0, 12, name.strip(), align="C")
        self.ln(2)

    def add_contact_line(self, contact: str) -> None:
        """Add contact info line (centered, smaller font)."""
        self.set_font(self.font_name, "", 10)
        self.set_x(self.l_margin)  # Reset to left margin
        self.set_text_color(64, 64, 64)
        # Use multi_cell for long contact lines that might wrap
        self.multi_cell(0, 6, contact.strip(), align="C")
        self.set_text_color(0, 0, 0)

    def add_section_header(self, title: str) -> None:
        """Add section header (H2 equivalent) with underline."""
        self.ln(6)
        self.set_font(self.font_name, "B", 16)
        self.set_x(self.l_margin)  # Ensure we start at left margin
        self.multi_cell(0, 10, title.upper())
        # Add subtle underline
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def add_experience_header(self, title: str, company: str, date_location: str) -> None:
        """Add experience entry header with title, company, and date/location.

        If title contains '|', renders part before '|' in bold, rest in regular.
        """
        self.set_x(self.l_margin)  # Ensure we start at left margin
        title_clean = title.strip()

        # Check if title contains '|' - split into bold and regular parts
        if "|" in title_clean:
            parts = title_clean.split("|", 1)
            bold_part = parts[0].strip()
            regular_part = "| " + parts[1].strip() if len(parts) > 1 else ""

            # Render bold part
            self.set_font(self.font_name, "B", 12)
            bold_width = self.get_string_width(bold_part + " ") + 2
            self.cell(bold_width, 7, bold_part + " ")

            # Render regular part (including '|') on same line
            if regular_part:
                self.set_font(self.font_name, "", 12)
                self.write(7, regular_part)
            self.ln(7)
        else:
            # No '|' - render entire title in bold
            self.set_font(self.font_name, "B", 12)
            self.multi_cell(0, 7, title_clean)

        # Company and date on same line (if both provided)
        if company or date_location:
            self.set_font(self.font_name, "", 10)
            self.set_x(self.l_margin)  # Reset position after multi_cell
            company_clean = company.strip() if company else ""
            date_clean = date_location.strip() if date_location else ""

            if company_clean and date_clean:
                # Both company and date - combine with separator
                combined = f"{company_clean} | {date_clean}"
                combined_width = self.get_string_width(combined) + 2
                available_width = self.w - self.l_margin - self.r_margin

                self.set_text_color(96, 96, 96)
                if combined_width < available_width:
                    # Fits on one line
                    self.multi_cell(0, 5, combined)
                else:
                    # Put on separate lines if too long
                    self.multi_cell(0, 5, company_clean)
                    self.set_x(self.l_margin)
                    self.multi_cell(0, 5, date_clean)
                self.set_text_color(0, 0, 0)
            elif company_clean:
                # Only company
                self.set_text_color(96, 96, 96)
                self.multi_cell(0, 5, company_clean)
                self.set_text_color(0, 0, 0)
            elif date_clean:
                # Only date
                self.set_text_color(96, 96, 96)
                self.multi_cell(0, 5, date_clean)
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

        # Ensure minimum width to prevent fpdf error
        if available_width < 20:
            self.ln()
            self.set_x(self.l_margin + bullet_indent + 5)
            available_width = self.w - self.r_margin - self.get_x()

        # Text with word wrap
        self._safe_multi_cell(available_width, 5, text.strip())

    def add_paragraph(self, text: str) -> None:
        """Add a regular paragraph."""
        self.set_font(self.font_name, "", 10)
        self.set_x(self.l_margin)  # Reset to left margin
        self.multi_cell(0, 5, text.strip())
        self.ln(2)

    def add_titled_paragraph(self, title: str, description: str) -> None:
        """Add a paragraph with bold title followed by regular description.

        Format: "**Title**: Description text" renders as bold title, regular text.
        Wraps to left margin if description is long.
        """
        self.set_x(self.l_margin)

        title_text = f"{title}: "
        desc_clean = description.strip()

        # Write title in bold - use cell for short title to ensure bold renders
        self.set_font(self.font_name, "B", 10)
        title_width = self.get_string_width(title_text) + 2
        self.cell(title_width, 5, title_text)

        # Write description in regular - use write() for natural wrapping
        self.set_font(self.font_name, "", 10)
        self.write(5, desc_clean)
        self.ln(6)  # Move to next line with slight spacing

    def add_skill_line(self, category: str, skills: str) -> None:
        """Add a skill category line (e.g., 'Languages: Python, TypeScript').

        Category is bold, followed by skills on the same line. If skills wrap,
        continuation lines start at left margin (not indented).
        """
        self.set_x(self.l_margin)

        cat_text = f"{category}: "
        skills_clean = skills.strip()

        # Write category in bold - use cell to ensure bold renders
        self.set_font(self.font_name, "B", 10)
        cat_width = self.get_string_width(cat_text) + 2
        self.cell(cat_width, 5, cat_text)

        # Write skills in regular - use write() for natural wrapping to left margin
        self.set_font(self.font_name, "", 10)
        self.write(5, skills_clean)
        self.ln(5)  # Move to next line after skills


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
    # Sanitize malformed markdown and unsupported characters before parsing
    markdown = _sanitize_markdown_bold(markdown)
    markdown = _sanitize_unsupported_chars(markdown)
    parsed = parse_markdown_cv(markdown)
    pdf = CVPDFGenerator()

    # Set PDF metadata for better indexing
    pdf.set_title(f"{parsed['name']} | CV" if parsed["name"] else "Tailored CV")
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


def _parse_experience_entries(content: list[str]) -> list[dict[str, Any]]:
    """Parse experience content into structured entries.

    Each entry has: title, company, date_location, bullets (list of strings)
    """
    entries: list[dict[str, Any]] = []
    current_entry: dict[str, Any] | None = None
    i = 0

    while i < len(content):
        line = content[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # H3 or bold line: likely job title - starts a new entry
        if line.startswith("### ") or (line.startswith("**") and line.endswith("**")):
            # Save previous entry
            if current_entry:
                entries.append(current_entry)

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
                    clean_line = re.sub(r"^\*\*|\*\*", "", next_line).strip()
                    parts = re.split(r"\s*[|•·]\s*", clean_line.strip("*_ "))
                    if len(parts) >= 1:
                        company = parts[0]
                    if len(parts) >= 2:
                        date_location = " | ".join(parts[1:])
                    i += 1

            current_entry = {
                "title": title,
                "company": company,
                "date_location": date_location,
                "bullets": [],
            }
            i += 1
            continue

        # Bullet points
        if line.startswith(("-", "*", "•")) and not line.startswith("**"):
            bullet_text = re.sub(r"^[-*•]\s*", "", line)
            bullet_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", bullet_text)
            bullet_text = re.sub(r"\*([^*]+)\*", r"\1", bullet_text)
            if current_entry:
                current_entry["bullets"].append(bullet_text)
            i += 1
            continue

        # Regular text - treat as a bullet if we have an entry
        if line and not line.startswith("#"):
            clean_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            clean_text = re.sub(r"\*([^*]+)\*", r"\1", clean_text)
            if current_entry:
                current_entry["bullets"].append(clean_text)

        i += 1

    # Don't forget the last entry
    if current_entry:
        entries.append(current_entry)

    return entries


def _estimate_entry_height(entry: dict[str, Any]) -> float:
    """Estimate the height of an experience entry in mm.

    This is approximate - used to decide if we need a page break.
    """
    height = 7.0  # Title line
    if entry["company"] or entry["date_location"]:
        height += 5.0  # Company/date line
    height += 1.0  # Spacing after header
    # Each bullet is approximately 5-10mm depending on wrap
    # Estimate 7mm per bullet as average
    height += len(entry["bullets"]) * 7.0
    return height


def _render_experience_section(pdf: CVPDFGenerator, content: list[str]) -> None:
    """Render experience section with job entries.

    Keeps each job entry together (no page breaks within an entry)
    and adds spacing between entries.
    """
    entries = _parse_experience_entries(content)
    first_entry = True

    for entry in entries:
        # Add spacing between entries (not before first)
        if not first_entry:
            pdf.ln(6)
        first_entry = False

        # Check if entry fits on current page, if not start new page
        entry_height = _estimate_entry_height(entry)
        space_left = pdf.h - pdf.get_y() - pdf.b_margin
        if entry_height > space_left and space_left < pdf.h * 0.5:
            # Entry won't fit and we're past halfway down the page - new page
            pdf.add_page()

        # Render the entry
        pdf.add_experience_header(entry["title"], entry["company"], entry["date_location"])
        for bullet in entry["bullets"]:
            pdf.add_bullet_point(bullet)


def _render_skills_section(pdf: CVPDFGenerator, content: list[str]) -> None:
    """Render skills section."""
    for line in content:
        line = line.strip()
        if not line:
            continue

        # Check for category: skills pattern (handles bold markers around category)
        # Matches patterns like: "Languages:", "**Languages:**", "*Languages:**"
        category_match = re.match(r"^[\*_]*([A-Za-z][A-Za-z &/]+?)[\*_]*:\s*(.*)$", line)
        if category_match and not line.startswith(("-", "•")):
            category = category_match.group(1).strip()
            skills = category_match.group(2).strip()
            # Clean any remaining bold markers from skills
            skills = re.sub(r"^\*+\s*", "", skills)  # Leading asterisks
            skills = re.sub(r"\*+$", "", skills)  # Trailing asterisks
            pdf.add_skill_line(category, skills)
        # Bullet point (but not bold markers **)
        elif (
            line.startswith("-")
            or line.startswith("•")
            or (line.startswith("*") and not line.startswith("**") and ":" not in line[:30])
        ):
            skill_text = re.sub(r"^[-*•]\s*", "", line)
            # Clean all markdown formatting
            skill_text = re.sub(r"\*+([^*]+)\*+", r"\1", skill_text)  # Any asterisk pattern
            skill_text = skill_text.strip("*_ ")
            pdf.add_bullet_point(skill_text)
        # Regular text
        elif line:
            clean = re.sub(r"\*+([^*]+)\*+", r"\1", line)  # Any asterisk pattern
            pdf.add_paragraph(clean.strip("*_ "))


def _render_education_section(pdf: CVPDFGenerator, content: list[str]) -> None:
    """Render education section."""
    i = 0
    first_entry = True
    while i < len(content):
        line = content[i].strip()

        if not line:
            i += 1
            continue

        # H3 or bold: degree/institution
        if line.startswith("### ") or (line.startswith("**") and line.endswith("**")):
            # Add spacing between education entries (not before first)
            if not first_entry:
                pdf.ln(4)
            first_entry = False

            title = re.sub(r"^###\s*", "", line)
            title = re.sub(r"^\*\*|\*\*$", "", title)

            institution = ""
            date_location = ""

            if i + 1 < len(content):
                next_line = content[i + 1].strip()
                # Check for institution line - may start with ** for bold
                # Don't skip lines starting with ** as those are institution names
                is_bullet = next_line.startswith(("-", "•")) or (
                    next_line.startswith("*") and not next_line.startswith("**")
                )
                if next_line and not next_line.startswith("#") and not is_bullet:
                    # Strip bold markers and parse
                    clean_line = re.sub(r"^\*\*|\*\*", "", next_line).strip()
                    parts = re.split(r"\s*[|•·]\s*", clean_line.strip("*_ "))
                    if len(parts) >= 1:
                        institution = parts[0]
                    if len(parts) >= 2:
                        date_location = " | ".join(parts[1:])
                    i += 1

            pdf.add_experience_header(title, institution, date_location)
            i += 1
            continue

        # Bullet points (but NOT bold markers **)
        if (
            line.startswith("-")
            or line.startswith("•")
            or (line.startswith("*") and not line.startswith("**"))
        ):
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
    """Render generic section (summary, certifications, projects, etc.)."""
    for line in content:
        line = line.strip()
        if not line:
            continue

        # Bullet points (but not bold **)
        if line.startswith(("-", "*", "•")) and not line.startswith("**"):
            text = re.sub(r"^[-*•]\s*", "", line)
            text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
            text = re.sub(r"\*([^*]+)\*", r"\1", text)
            pdf.add_bullet_point(text)
        # Bold title pattern: **Title**: Description
        elif line.startswith("**"):
            # Match **Title**: Description pattern
            title_match = re.match(r"^\*\*([^*]+)\*\*:\s*(.*)$", line)
            if title_match:
                title = title_match.group(1).strip()
                description = title_match.group(2).strip()
                pdf.add_titled_paragraph(title, description)
            else:
                # Just bold text without colon - strip and render as paragraph
                clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
                pdf.add_paragraph(clean)
        # Regular text
        elif not line.startswith("#"):
            clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
            pdf.add_paragraph(clean)
