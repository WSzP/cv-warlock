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

Supported styles:
- plain: Classic, clean CV layout (original style)
- modern: Contemporary design with accent colors and refined visual hierarchy
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from fpdf import FPDF, ViewerPreferences  # type: ignore[import-untyped]


class CVStyle(str, Enum):
    """Available CV PDF styles."""

    PLAIN = "plain"
    MODERN = "modern"


@dataclass
class StyleConfig:
    """Configuration for a CV style."""

    # Colors (RGB tuples)
    accent_color: tuple[int, int, int]
    accent_dark: tuple[int, int, int]  # Darker shade for gradients/hover
    accent_light: tuple[int, int, int]  # Light tint for backgrounds
    text_primary: tuple[int, int, int]
    text_secondary: tuple[int, int, int]
    text_muted: tuple[int, int, int]
    text_on_accent: tuple[int, int, int]  # Text color on accent backgrounds
    divider_color: tuple[int, int, int]
    card_background: tuple[int, int, int]  # Subtle background for cards

    # Typography
    name_size: int
    section_header_size: int
    job_title_size: int
    body_size: int
    contact_size: int

    # Layout
    left_margin: float
    top_margin: float
    right_margin: float
    section_spacing: float
    entry_spacing: float

    # Style features
    use_header_band: bool  # Full-width colored header
    header_band_height: float
    use_accent_bar: bool  # Left side accent bar
    accent_bar_width: float
    section_header_uppercase: bool
    section_header_underline: bool
    section_header_band: bool  # Full colored background for section headers
    use_skill_pills: bool  # Display skills as pill badges
    use_entry_cards: bool  # Subtle background for job entries
    contact_separator: str


# Style presets
STYLE_CONFIGS: dict[CVStyle, StyleConfig] = {
    CVStyle.PLAIN: StyleConfig(
        # Colors - all grayscale for plain
        accent_color=(0, 0, 0),
        accent_dark=(0, 0, 0),
        accent_light=(240, 240, 240),
        text_primary=(0, 0, 0),
        text_secondary=(64, 64, 64),
        text_muted=(96, 96, 96),
        text_on_accent=(255, 255, 255),
        divider_color=(200, 200, 200),
        card_background=(250, 250, 250),
        # Typography
        name_size=18,
        section_header_size=16,
        job_title_size=12,
        body_size=10,
        contact_size=10,
        # Layout
        left_margin=20.0,
        top_margin=20.0,
        right_margin=20.0,
        section_spacing=6.0,
        entry_spacing=6.0,
        # Style features
        use_header_band=False,
        header_band_height=0.0,
        use_accent_bar=False,
        accent_bar_width=0.0,
        section_header_uppercase=True,
        section_header_underline=True,
        section_header_band=False,
        use_skill_pills=False,
        use_entry_cards=False,
        contact_separator=" | ",
    ),
    CVStyle.MODERN: StyleConfig(
        # Colors - sophisticated deep navy palette
        accent_color=(20, 50, 90),  # Deep navy blue
        accent_dark=(15, 35, 65),  # Darker navy for depth
        accent_light=(235, 242, 250),  # Very light blue tint
        text_primary=(25, 30, 38),  # Near-black with warmth
        text_secondary=(55, 65, 80),  # Dark slate
        text_muted=(90, 100, 115),  # Medium slate
        text_on_accent=(255, 255, 255),  # White on accent
        divider_color=(220, 228, 238),  # Subtle blue-gray divider
        card_background=(247, 250, 253),  # Very subtle blue tint
        # Typography - larger, bolder
        name_size=26,
        section_header_size=11,
        job_title_size=11,
        body_size=10,
        contact_size=9,
        # Layout - generous spacing
        left_margin=20.0,
        top_margin=15.0,
        right_margin=20.0,
        section_spacing=10.0,
        entry_spacing=8.0,
        # Style features
        use_header_band=True,
        header_band_height=38.0,
        use_accent_bar=False,
        accent_bar_width=0.0,
        section_header_uppercase=True,
        section_header_underline=False,
        section_header_band=True,
        use_skill_pills=True,
        use_entry_cards=False,
        contact_separator="  \u2022  ",
    ),
}


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
    """Replace characters not well-supported across fonts with ASCII equivalents.

    Most Unicode symbols (arrows, math symbols) are handled by the DejaVu Sans
    fallback font. Only replace characters that cause rendering issues.
    """
    replacements = {
        # Curly quotes to straight quotes (stylistic consistency)
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

    Supports multiple visual styles via the `style` parameter.
    """

    def __init__(self, style: CVStyle = CVStyle.PLAIN) -> None:
        super().__init__(format="A4")
        self.style = style
        self.config = STYLE_CONFIGS[style]

        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(
            left=self.config.left_margin,
            top=self.config.top_margin,
            right=self.config.right_margin,
        )

        # Track content area offset for accent bar
        self._content_offset = self.config.accent_bar_width + 2 if self.config.use_accent_bar else 0

        # Load Poppins font for professional, modern look
        self._setup_poppins_font()

    def _safe_multi_cell(self, w: float, h: float, text: str, **kwargs: object) -> None:
        """Multi-cell with width validation to prevent fpdf errors."""
        # Default to left alignment if not specified
        if "align" not in kwargs:
            kwargs["align"] = "L"

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
        """Set up Poppins font with DejaVu Sans fallback for Unicode symbols."""
        fonts = _get_poppins_fonts()

        if fonts:
            # Add Poppins with proper bold/italic variants
            self.add_font("Poppins", "", str(fonts["regular"]))
            self.add_font("Poppins", "B", str(fonts["bold"]))
            self.add_font("Poppins", "I", str(fonts["italic"]))
            self.add_font("Poppins", "BI", str(fonts["bold_italic"]))
            self.font_name = "Poppins"

            # Add DejaVu Sans as fallback for Unicode symbols (arrows, math, etc.)
            dejavu_path = _FONTS_DIR / "DejaVuSans.ttf"
            if dejavu_path.exists():
                self.add_font("DejaVuSans", "", str(dejavu_path))
                self.set_fallback_fonts(["DejaVuSans"])
        else:
            # Fallback to built-in Helvetica (limited character support)
            self.font_name = "Helvetica"

    def header(self) -> None:
        """Draw page header elements."""
        # Draw accent bar on left side if enabled (not used in new modern design)
        if self.config.use_accent_bar:
            self.set_fill_color(*self.config.accent_color)
            self.rect(x=0, y=0, w=self.config.accent_bar_width, h=self.h, style="F")

    def footer(self) -> None:
        """Add page number footer for multi-page CVs."""
        if self.page_no() > 1:
            if self.style == CVStyle.MODERN:
                # Modern: elegant pill-style page number aligned right
                self.set_y(-12)
                page_text = str(self.page_no())

                # Calculate pill dimensions
                self.set_font(self.font_name, "B", 8)
                text_width = self.get_string_width(page_text)
                pill_width = text_width + 10
                pill_height = 6
                pill_x = self.w - self.r_margin - pill_width

                # Draw pill background
                self.set_fill_color(*self.config.accent_color)
                self.rect(pill_x, self.get_y(), pill_width, pill_height, style="F")

                # Draw page number in white
                self.set_xy(pill_x, self.get_y() + 0.8)
                self.set_text_color(*self.config.text_on_accent)
                self.cell(pill_width, pill_height - 1, page_text, align="C")
                self.set_text_color(*self.config.text_primary)
            else:
                # Plain: simple centered page number
                self.set_y(-15)
                self.set_font(self.font_name, "I", 8)
                self.set_text_color(*self.config.text_muted)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")
                self.set_text_color(*self.config.text_primary)

    def _draw_header_band(self, name: str) -> None:
        """Draw the full-width header band with name for modern style."""
        band_height = self.config.header_band_height

        # Draw the colored header band
        self.set_fill_color(*self.config.accent_color)
        self.rect(x=0, y=0, w=self.w, h=band_height, style="F")

        # Add subtle gradient effect with darker bottom edge
        self.set_fill_color(*self.config.accent_dark)
        self.rect(x=0, y=band_height - 2, w=self.w, h=2, style="F")

        # Position name in lower portion of header band for better visual balance
        # Font size 26pt ≈ 9mm, position so name sits ~8mm from bottom
        name_y = band_height - 17
        self.set_xy(self.l_margin, name_y)
        self.set_font(self.font_name, "B", self.config.name_size)
        self.set_text_color(*self.config.text_on_accent)
        self.cell(0, 10, name.strip(), align="L")

        # Move cursor below the header band
        self.set_y(band_height + 5)
        self.set_text_color(*self.config.text_primary)

    def add_name(self, name: str) -> None:
        """Add candidate name as main heading (H1 equivalent)."""
        if self.style == CVStyle.MODERN and self.config.use_header_band:
            # Modern: draw full header band with name
            self._draw_header_band(name)
        elif self.style == CVStyle.MODERN:
            # Modern without header band: accent colored name
            self.set_font(self.font_name, "B", self.config.name_size)
            self.set_x(self.l_margin)
            self.set_text_color(*self.config.accent_color)
            self.multi_cell(0, 12, name.strip(), align="L")
            self.set_text_color(*self.config.text_primary)
            self.ln(1)
        else:
            # Plain: centered black name
            self.set_font(self.font_name, "B", self.config.name_size)
            self.set_x(self.l_margin)
            self.set_text_color(*self.config.text_primary)
            self.multi_cell(0, 12, name.strip(), align="C")
            self.ln(2)

    def add_contact_line(self, contact: str, links: list[tuple[str, str]] | None = None) -> None:
        """Add contact info line with optional clickable links.

        Args:
            contact: The contact text to display.
            links: List of (display_text, url) tuples for clickable links.
        """
        self.set_font(self.font_name, "", self.config.contact_size)
        self.set_x(self.l_margin)
        self.set_text_color(*self.config.text_secondary)

        if links:
            # Render contact line with clickable links
            self._render_contact_with_links(contact, links)
        else:
            # Simple text rendering
            if self.style == CVStyle.MODERN:
                self.multi_cell(0, 5, contact.strip(), align="L")
            else:
                self.multi_cell(0, 6, contact.strip(), align="C")

        self.set_text_color(*self.config.text_primary)

    def _render_contact_with_links(self, contact: str, links: list[tuple[str, str]]) -> None:
        """Render contact line with clickable links inline."""
        line_height = 5 if self.style == CVStyle.MODERN else 6
        remaining = contact

        # Process each link in order of appearance
        for display_text, url in links:
            if display_text not in remaining:
                continue

            # Split at this link
            before, after = remaining.split(display_text, 1)

            # Render text before the link
            if before:
                self.write(line_height, before)

            # Render the link (clickable, accent colored)
            self.set_text_color(*self.config.accent_color)
            self.write(line_height, display_text, url)
            self.set_text_color(*self.config.text_secondary)

            remaining = after

        # Render any remaining text after the last link
        if remaining:
            self.write(line_height, remaining)

        self.ln(line_height)

    def add_section_header(self, title: str) -> None:
        """Add section header (H2 equivalent)."""
        self.ln(self.config.section_spacing)

        # Check if there's enough space for the header + minimum content
        # If not, start a new page to avoid orphaned headers
        min_content_height = 35  # Header + at least one entry or paragraph
        space_left = self.h - self.get_y() - self.b_margin
        if space_left < min_content_height:
            self.add_page()

        # Apply uppercase if configured
        display_title = title.upper() if self.config.section_header_uppercase else title

        if self.style == CVStyle.MODERN and self.config.section_header_band:
            # Modern: bold text with thick accent underline
            start_y = self.get_y()

            # Draw the section title in bold navy
            self.set_xy(self.l_margin, start_y)
            self.set_font(self.font_name, "B", self.config.section_header_size + 2)
            self.set_text_color(*self.config.accent_color)
            title_width = self.get_string_width(display_title)
            self.cell(title_width + 2, 7, display_title, align="L")

            # Draw thick accent underline directly under the text
            underline_y = start_y + 8
            self.set_fill_color(*self.config.accent_color)
            self.rect(
                x=self.l_margin,
                y=underline_y,
                w=title_width + 4,
                h=1.5,
                style="F",
            )

            # Move below the header
            self.set_y(underline_y + 6)
            self.set_text_color(*self.config.text_primary)

        elif self.style == CVStyle.MODERN:
            # Modern without band: accent color with marker
            self.set_font(self.font_name, "B", self.config.section_header_size)
            self.set_x(self.l_margin)
            self.set_text_color(*self.config.accent_color)

            marker_y = self.get_y() + 3
            self.set_fill_color(*self.config.accent_color)
            self.rect(self.l_margin - 6, marker_y, 2, 6, style="F")

            self.multi_cell(0, 8, display_title, align="L")
            self.set_text_color(*self.config.text_primary)
            self.ln(2)
        else:
            # Plain: black header with underline
            self.set_font(self.font_name, "B", self.config.section_header_size)
            self.set_x(self.l_margin)
            self.set_text_color(*self.config.text_primary)
            self.multi_cell(0, 10, display_title, align="L")

            if self.config.section_header_underline:
                self.set_draw_color(*self.config.divider_color)
                self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())

            self.ln(3)

    def add_experience_header(self, title: str, company: str, date_location: str) -> None:
        """Add experience entry header with title, company, and date/location.

        If title contains '|', renders part before '|' in bold, rest in regular.
        """
        self.set_x(self.l_margin)
        title_clean = title.strip()
        title_size = self.config.job_title_size

        # Check if title contains '|' - split into bold and regular parts
        if "|" in title_clean:
            parts = title_clean.split("|", 1)
            bold_part = parts[0].strip()
            regular_part = "| " + parts[1].strip() if len(parts) > 1 else ""

            # Render bold part
            self.set_font(self.font_name, "B", title_size)
            self.set_text_color(*self.config.text_primary)
            bold_width = self.get_string_width(bold_part + " ") + 2
            self.cell(bold_width, 7, bold_part + " ")

            # Render regular part (including '|') on same line
            if regular_part:
                self.set_font(self.font_name, "", title_size)
                self.set_text_color(*self.config.text_secondary)
                self.write(7, regular_part)
                self.set_text_color(*self.config.text_primary)
            self.ln(7)
        else:
            # No '|' - render entire title in bold
            self.set_font(self.font_name, "B", title_size)
            self.set_text_color(*self.config.text_primary)
            self.multi_cell(0, 7, title_clean, align="L")

        # Company and date on same line (if both provided)
        if company or date_location:
            self.set_font(self.font_name, "", self.config.body_size)
            self.set_x(self.l_margin)
            company_clean = company.strip() if company else ""
            date_clean = date_location.strip() if date_location else ""

            if company_clean and date_clean:
                combined = f"{company_clean} | {date_clean}"
                combined_width = self.get_string_width(combined) + 2
                available_width = self.w - self.l_margin - self.r_margin

                self.set_text_color(*self.config.text_muted)
                if combined_width < available_width:
                    self.multi_cell(0, 5, combined, align="L")
                else:
                    self.multi_cell(0, 5, company_clean, align="L")
                    self.set_x(self.l_margin)
                    self.multi_cell(0, 5, date_clean, align="L")
                self.set_text_color(*self.config.text_primary)
            elif company_clean:
                self.set_text_color(*self.config.text_muted)
                self.multi_cell(0, 5, company_clean, align="L")
                self.set_text_color(*self.config.text_primary)
            elif date_clean:
                self.set_text_color(*self.config.text_muted)
                self.multi_cell(0, 5, date_clean, align="L")
                self.set_text_color(*self.config.text_primary)

        self.ln(1)

    def add_bullet_point(self, text: str, indent: int = 0) -> None:
        """Add a bullet point with proper formatting."""
        self.set_font(self.font_name, "", self.config.body_size)
        self.set_text_color(*self.config.text_primary)
        self.set_x(self.l_margin)

        bullet_indent = 5 + (indent * 5)

        # Modern style: accent-colored bullet
        if self.style == CVStyle.MODERN:
            self.cell(bullet_indent, 5, "")
            self.set_text_color(*self.config.accent_color)
            self.cell(5, 5, "\u2022")  # Bullet character
            self.set_text_color(*self.config.text_primary)
        else:
            self.cell(bullet_indent, 5, "")
            self.cell(5, 5, "\u2022")

        # Calculate available width for text
        text_start_x = self.get_x()
        available_width = self.w - self.r_margin - text_start_x

        if available_width < 20:
            self.ln()
            self.set_x(self.l_margin + bullet_indent + 5)
            available_width = self.w - self.r_margin - self.get_x()

        self._safe_multi_cell(available_width, 5, text.strip())

    def add_paragraph(self, text: str) -> None:
        """Add a regular paragraph."""
        self.set_font(self.font_name, "", self.config.body_size)
        self.set_text_color(*self.config.text_primary)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5, text.strip(), align="L")
        self.ln(2)

    def add_titled_paragraph(self, title: str, description: str) -> None:
        """Add a paragraph with bold title followed by regular description.

        Format: "**Title**: Description text" renders as bold title, regular text.
        Wraps to left margin if description is long.
        """
        self.set_x(self.l_margin)

        title_text = f"{title}: "
        desc_clean = description.strip()

        # Write title in bold
        self.set_font(self.font_name, "B", self.config.body_size)
        if self.style == CVStyle.MODERN:
            self.set_text_color(*self.config.accent_color)
        else:
            self.set_text_color(*self.config.text_primary)
        title_width = self.get_string_width(title_text) + 2
        self.cell(title_width, 5, title_text)

        # Write description in regular
        self.set_font(self.font_name, "", self.config.body_size)
        self.set_text_color(*self.config.text_primary)
        self.write(5, desc_clean)
        self.ln(6)

    def _draw_skill_pill(self, skill: str, x: float, y: float) -> float:
        """Draw a single skill pill badge and return its width."""
        self.set_font(self.font_name, "", 8)
        text_width = self.get_string_width(skill)
        pill_width = text_width + 6  # Padding
        pill_height = 5.5

        # Draw pill background
        self.set_fill_color(*self.config.accent_light)
        # Draw rounded rectangle (approximate with rect since fpdf2 doesn't have round_rect easily)
        self.rect(x, y, pill_width, pill_height, style="F")

        # Draw text
        self.set_xy(x + 3, y + 0.8)
        self.set_text_color(*self.config.accent_color)
        self.cell(text_width, 4, skill)

        return pill_width

    def add_skill_line(self, category: str, skills: str) -> None:
        """Add a skill category line (e.g., 'Languages: Python, TypeScript').

        Category is bold, followed by skills on the same line. If skills wrap,
        continuation lines start at left margin (not indented).
        """
        if self.style == CVStyle.MODERN and self.config.use_skill_pills:
            # Modern with pills: category header then skill pills
            self._add_skill_pills(category, skills)
        else:
            # Standard rendering
            self.set_x(self.l_margin)

            cat_text = f"{category}: "
            skills_clean = skills.strip()

            # Write category in bold
            self.set_font(self.font_name, "B", self.config.body_size)
            if self.style == CVStyle.MODERN:
                self.set_text_color(*self.config.accent_color)
            else:
                self.set_text_color(*self.config.text_primary)
            cat_width = self.get_string_width(cat_text) + 2
            self.cell(cat_width, 5, cat_text)

            # Write skills in regular
            self.set_font(self.font_name, "", self.config.body_size)
            self.set_text_color(*self.config.text_primary)
            self.write(5, skills_clean)
            self.ln(5)

    def _add_skill_pills(self, category: str, skills: str) -> None:
        """Add skills as pill badges with category header."""
        # Category header
        self.set_x(self.l_margin)
        self.set_font(self.font_name, "B", 9)
        self.set_text_color(*self.config.accent_color)
        self.cell(0, 6, category)
        self.ln(6)

        # Parse skills (comma or semicolon separated)
        skill_list = [s.strip() for s in skills.replace(";", ",").split(",") if s.strip()]

        # Layout pills with wrapping
        x = self.l_margin
        y = self.get_y()
        max_x = self.w - self.r_margin
        pill_spacing = 3
        row_height = 7

        for skill in skill_list:
            self.set_font(self.font_name, "", 8)
            pill_width = self.get_string_width(skill) + 6

            # Check if we need to wrap to next line
            if x + pill_width > max_x:
                x = self.l_margin
                y += row_height

            # Draw the pill
            self._draw_skill_pill(skill, x, y)
            x += pill_width + pill_spacing

        # Move cursor below all pills
        self.set_y(y + row_height + 2)
        self.set_text_color(*self.config.text_primary)


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


def generate_cv_pdf(markdown: str, style: CVStyle | str = CVStyle.PLAIN) -> bytes:
    """Generate a well-structured PDF from markdown CV.

    Args:
        markdown: The CV content in markdown format.
        style: The visual style to use ('plain' or 'modern'). Default is 'plain'.

    Returns:
        PDF content as bytes.
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
        # Matches patterns like: "Languages:", "**Languages:**", "*Languages:**", "High-Growth SaaS:"
        category_match = re.match(r"^[\*_]*([A-Za-z][A-Za-z0-9 &/\-]+?)[\*_]*:\s*(.*)$", line)
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
        # Bold title pattern: **Title:** Description OR **Title**: Description
        elif line.startswith("**"):
            # Match **Title:** Description (colon inside) or **Title**: Description (colon outside)
            title_match = re.match(r"^\*\*([^*]+?)(?::\*\*|\*\*:)\s*(.*)$", line)
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
