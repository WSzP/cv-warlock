"""Section rendering functions for CV PDF generation.

Each renderer handles a specific CV section type (experience, skills,
education, publications, generic) by calling methods on CVPDFGenerator.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from app.utils.pdf_generator.styles import CVStyle

if TYPE_CHECKING:
    from app.utils.pdf_generator.generator import CVPDFGenerator


def _render_section_content(pdf: CVPDFGenerator, header: str, content: list[str]) -> None:
    """Render section content with appropriate formatting based on section type.

    Supports both English and Hungarian section names.
    """
    header_lower = header.lower()

    # Experience/Work sections: parse job entries
    # Hungarian: "tapasztalat" = experience, "onkentes" = volunteer
    if any(
        kw in header_lower
        for kw in ["experience", "work", "employment", "history", "tapasztalat", "önkéntes"]
    ):
        _render_experience_section(pdf, content)
    # Skills sections: render as category lists
    # Hungarian: "keszseg" = skill, "kompetencia" = competence
    elif any(
        kw in header_lower for kw in ["skill", "technical", "competenc", "készség", "kompetencia"]
    ):
        _render_skills_section(pdf, content)
    # Education: similar to experience but simpler
    # Hungarian: "tanulmany" = studies, "vegzettseg" = qualification, "kepzes" = training
    elif any(
        kw in header_lower
        for kw in [
            "education",
            "academic",
            "qualification",
            "tanulmány",
            "végzettség",
            "képzés",
        ]
    ):
        _render_education_section(pdf, content)
    # Publications: special handling for books/papers with URLs
    # Hungarian: "konyv" = book, "publikacio" = publication
    elif any(
        kw in header_lower
        for kw in ["publication", "book", "paper", "article", "könyv", "publikáció"]
    ):
        _render_publications_section(pdf, content)
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

            # Next non-empty line might be company/date
            next_idx = i + 1
            while next_idx < len(content) and not content[next_idx].strip():
                next_idx += 1

            if next_idx < len(content):
                next_line = content[next_idx].strip()
                # Check if it's italic (company) or contains date patterns
                if next_line.startswith("*") or re.search(r"\d{4}", next_line):
                    # Parse company | location | date pattern
                    clean_line = re.sub(r"^\*\*|\*\*", "", next_line).strip()
                    parts = re.split(r"\s*[|•·]\s*", clean_line.strip("*_ "))
                    if len(parts) >= 1:
                        company = parts[0]
                    if len(parts) >= 2:
                        date_location = " | ".join(parts[1:])
                    i = next_idx

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
        # Matches patterns like: "Languages:", "**Languages:**", "*Languages:**", "High-Growth SaaS:"
        # Uses Unicode \w to support accented characters (e.g., "Programozasi nyelvek:")
        category_match = re.match(r"^[\*_]*([\w][\w &/\-]+?)[\*_]*:\s*(.*)$", line, re.UNICODE)
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

            # Next non-empty line might be institution/date
            next_idx = i + 1
            while next_idx < len(content) and not content[next_idx].strip():
                next_idx += 1

            if next_idx < len(content):
                next_line = content[next_idx].strip()
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
                    i = next_idx

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


def _render_publications_section(pdf: CVPDFGenerator, content: list[str]) -> None:
    """Render publications section with proper formatting for books/papers.

    Expected format: - Title (Publisher, Year): URL
    Renders as: Title with publisher/year, clickable link on separate line.
    """
    for line in content:
        line = line.strip()
        if not line:
            continue

        # Skip non-bullet lines
        if not line.startswith(("-", "*", "•")):
            # Regular text - render as paragraph
            clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            pdf.add_paragraph(clean)
            continue

        # Remove bullet marker
        text = re.sub(r"^[-*•]\s*", "", line)

        # Parse publication format: Title (Publisher, Year): URL
        # or: Title (Publisher, Year): [Link Text](URL)
        pub_match = re.match(
            r"^(.+?)\s*\(([^)]+)\)(?::\s*(?:\[([^\]]+)\]\(([^)]+)\)|(\S+)))?\s*$",
            text,
        )

        if pub_match:
            title = pub_match.group(1).strip()
            publisher_info = pub_match.group(2).strip()  # "Publisher, Year"
            link_text = pub_match.group(3)  # From markdown link
            link_url = pub_match.group(4)  # From markdown link
            plain_url = pub_match.group(5)  # Plain URL

            # Determine URL and display text
            url = link_url or plain_url
            display_url = link_text or (
                plain_url[:50] + "..." if plain_url and len(plain_url) > 50 else plain_url
            )

            # Render title in bold with publisher info
            pdf.set_x(pdf.l_margin)
            pdf.set_font(pdf.font_name, "B", pdf.config.body_size)
            if pdf.style == CVStyle.MODERN:
                pdf.set_text_color(*pdf.config.accent_color)
            else:
                pdf.set_text_color(*pdf.config.text_primary)

            title_text = f"{title} ({publisher_info})"
            pdf.multi_cell(0, 5, title_text, align="L")

            # Render URL as clickable link on next line (indented)
            if url:
                pdf.set_x(pdf.l_margin + 5)
                pdf.set_font(pdf.font_name, "", pdf.config.body_size - 1)
                pdf._write_link(display_url or url, url, 4)
                pdf.set_text_color(*pdf.config.text_primary)
                pdf.ln(5)
            else:
                pdf.ln(1)
        else:
            # Fallback: render as simple bullet with link handling
            pdf.add_bullet_point(text)


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
        # Bold title pattern: **Title:** Description OR **Title**: Description OR **Title** Description
        elif line.startswith("**"):
            # Match **Title:** Description (colon inside) or **Title**: Description (colon outside)
            title_match = re.match(r"^\*\*([^*]+?)(?::\*\*|\*\*:)\s*(.*)$", line)
            if title_match:
                title = title_match.group(1).strip()
                description = title_match.group(2).strip()
                pdf.add_titled_paragraph(title, description)
            else:
                # Match **Title** Description (bold followed by regular text, no colon)
                inline_match = re.match(r"^\*\*([^*]+)\*\*\s+(.+)$", line)
                if inline_match:
                    title = inline_match.group(1).strip()
                    description = inline_match.group(2).strip()
                    pdf.add_inline_bold_text(title, description)
                else:
                    # Standalone bold text (no colon) - render as bold title
                    clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
                    pdf.add_bold_title(clean)
        # Regular text
        elif not line.startswith("#"):
            clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
            pdf.add_paragraph(clean)
