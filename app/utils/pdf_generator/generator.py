"""Core PDF generator class for CV documents.

Contains the CVPDFGenerator (FPDF subclass) with all primitive rendering
methods: name, contact, section headers, experience entries, bullets,
paragraphs, skills, and page header/footer.
"""

import re
from pathlib import Path

from fpdf import FPDF  # type: ignore[import-untyped]

from app.utils.pdf_generator.logos import (
    LogoRun,
    TitleRun,
    extract_logo_token,
    logo_aspect_ratio,
    logo_baseline_ratio,
    plan_title_runs,
)
from app.utils.pdf_generator.styles import STYLE_CONFIGS, CVStyle

# Get the fonts directory (relative to project root)
_FONTS_DIR = Path(__file__).parent.parent.parent.parent / "fonts"


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
    support including Romanian diacritics (t, s, a, a, i), accented characters,
    and other special symbols.

    Supports multiple visual styles via the `style` parameter.
    """

    def __init__(self, style: CVStyle = CVStyle.MODERN) -> None:
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

    def _write_link(self, text: str, url: str, line_height: float = 5) -> None:
        """Write a clickable link in distinct link color.

        Uses link_color (#0044CC) to indicate clickability.
        """
        self.set_text_color(*self.config.link_color)
        self.write(line_height, text, url)

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
                # Modern: page number in blue band that extends to bottom edge
                page_text = str(self.page_no())

                # Calculate dimensions - band extends from top of text area to page bottom
                self.set_font(self.font_name, "B", 10)
                text_width = self.get_string_width(page_text)
                band_width = text_width + 16
                band_height = 12  # Height of the visible band portion
                band_x = self.w - self.r_margin - band_width
                band_y = self.h - band_height  # Position at bottom edge

                # Draw blue band that extends to bottom of page
                self.set_fill_color(*self.config.accent_color)
                self.rect(band_x, band_y, band_width, band_height, style="F")

                # Draw page number in white, vertically centered in the band
                self.set_xy(band_x, band_y + 2.5)
                self.set_text_color(*self.config.text_on_accent)
                self.cell(band_width, band_height - 5, page_text, align="C")
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
        # Font size 26pt ~ 9mm, position so name sits ~8mm from bottom
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

            # Render the link (clickable, distinct link color with dotted underline)
            self._write_link(display_text, url, line_height)
            self.set_text_color(*self.config.text_secondary)

            remaining = after

        # Render any remaining text after the last link
        if remaining:
            self.write(line_height, remaining)

        self.ln(line_height)

    def _has_links(self, text: str) -> bool:
        """Check if text contains markdown or angle bracket links."""
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)|<(https?://[^>]+)>"
        return bool(re.search(link_pattern, text))

    def _render_text_with_links(self, text: str, line_height: float = 5) -> None:
        """Render text with inline clickable links.

        Supports both markdown links [text](url) and angle bracket links <url>.
        """
        # Find all links: markdown [text](url) and angle bracket <url>
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)|<(https?://[^>]+)>"

        last_end = 0

        for match in re.finditer(link_pattern, text):
            # Text before this link
            before = text[last_end : match.start()]
            if before:
                self.write(line_height, before)

            # Determine link text and URL
            if match.group(1) and match.group(2):
                # Markdown link [text](url)
                display_text = match.group(1)
                url = match.group(2)
            else:
                # Angle bracket link <url>
                url = match.group(3)
                display_text = url

            # Render the clickable link with distinct color and dotted underline
            self._write_link(display_text, url, line_height)
            self.set_text_color(*self.config.text_primary)

            last_end = match.end()

        # Render any remaining text after the last link
        if last_end < len(text):
            self.write(line_height, text[last_end:])

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
            # Modern: horizontal accent bar leading into bold text (same height)
            start_y = self.get_y()

            # Set up font to calculate text height
            self.set_font(self.font_name, "B", self.config.section_header_size + 2)
            self.set_text_color(*self.config.accent_color)

            # Draw horizontal accent bar from page edge to just before the text
            # Bar is vertically centered with the text
            bar_height = 4.5
            bar_y = start_y + 2.5  # Vertically center with text
            bar_end_x = self.l_margin - 3  # Small gap before text

            self.set_fill_color(*self.config.accent_color)
            self.rect(
                x=0,
                y=bar_y,
                w=bar_end_x,
                h=bar_height,
                style="F",
            )

            # Draw the section title in bold navy after the bar
            self.set_xy(self.l_margin, start_y)
            self.cell(0, 10, display_title, align="L")

            # Move below the header
            self.set_y(start_y + 12)
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

        If title contains '|' or en-dash, renders part before separator in bold, rest in regular.
        """
        self.set_x(self.l_margin)
        title_clean = title.strip()
        title_size = self.config.job_title_size

        # Company logo, from a markdown image token or the company registry
        runs = plan_title_runs(title_clean)
        if any(isinstance(run, LogoRun) for run in runs):
            self._render_title_runs(runs, title_size)
            self._render_company_line(company, date_location)
            return

        # No logo, but a broken token must never print as text
        title_clean, _, _ = extract_logo_token(title_clean)

        # Detect separator: pipe '|' or en-dash
        separator = None
        if "|" in title_clean:
            separator = "|"
        elif "\u2013" in title_clean:  # en-dash
            separator = "\u2013"

        # Check if title contains separator - split into bold and regular parts
        if separator:
            parts = title_clean.split(separator, 1)
            bold_part = parts[0].strip()
            regular_part = separator + " " + parts[1].strip() if len(parts) > 1 else ""

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
            # No separator - render entire title in bold
            self.set_font(self.font_name, "B", title_size)
            self.set_text_color(*self.config.text_primary)
            self.multi_cell(0, 7, title_clean, align="L")

        self._render_company_line(company, date_location)

    def _render_company_line(self, company: str, date_location: str) -> None:
        """Render the company and date/location line under an entry title."""
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

    def _render_title_runs(self, runs: list[TitleRun], title_size: int) -> None:
        """Render a title as inline text and logo runs on one line."""
        line_height = 7.0
        self.set_font(self.font_name, "B", title_size)

        for run in runs:
            if isinstance(run, LogoRun):
                self._draw_inline_logo(run.path, line_height)
                continue

            self.set_font(self.font_name, "B" if run.bold else "", title_size)
            self.set_text_color(
                *(self.config.text_primary if run.bold else self.config.text_secondary)
            )
            self.write(line_height, run.text)

        self.set_text_color(*self.config.text_primary)
        self.ln(line_height)

    def _draw_inline_logo(self, path: Path, line_height: float) -> None:
        """Draw a vector logo on the current line, sized by height.

        Width comes from the SVG viewBox, so the file's aspect ratio is
        preserved, and the logo's lettering rests on the text baseline
        rather than its outer box, so a padded logo does not float. Gaps
        scale with
        the logo: a solid block needs more air than a letter does, so the
        leading gap is wide, while the run after it usually starts with a
        space of its own, so the trailing gap is narrow.

        A logo that would cross the right margin is skipped rather than
        bleeding into it, and an unreadable file is skipped too: a CV must
        never fail to render over a logo.
        """
        height = self.config.inline_logo_height

        try:
            width = height * logo_aspect_ratio(path)
        except Exception:
            return

        x = self.get_x() + height * 0.6
        if x + width > self.w - self.r_margin:
            return

        # fpdf2 puts the text baseline at 0.5 * line height + 0.3 * font size
        baseline = self.get_y() + line_height / 2 + 0.3 * self.font_size

        try:
            self.image(str(path), x=x, y=baseline - height * logo_baseline_ratio(path), h=height)
        except Exception:
            return

        self.set_x(x + width + height * 0.2)

    def add_bullet_point(self, text: str, indent: int = 0) -> None:
        """Add a bullet point with proper formatting.

        Supports clickable links in markdown [text](url) or angle bracket <url> format.
        """
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

        # Check if text contains links - render link on new line below text
        if self._has_links(text):
            # Extract links and text separately
            link_pattern = r"\[([^\]]+)\]\(([^)]+)\)|<(https?://[^>]+)>"
            matches = list(re.finditer(link_pattern, text))

            if matches:
                # Get text before the first link
                first_match = matches[0]
                text_before = text[: first_match.start()].strip()

                # Render the text part
                if text_before:
                    self._safe_multi_cell(available_width, 5, text_before)
                else:
                    self.ln(5)

                # Render each link on its own indented line
                link_indent = self.l_margin + bullet_indent + 5
                for match in matches:
                    if match.group(1) and match.group(2):
                        # Markdown link [text](url) - show text as link
                        display_text = match.group(1)
                        url = match.group(2)
                    else:
                        # Angle bracket link <url>
                        url = match.group(3)
                        display_text = url

                    self.set_x(link_indent)
                    self._write_link(display_text, url, 5)
                    self.set_text_color(*self.config.text_primary)
                    self.ln(5)
            else:
                self._safe_multi_cell(available_width, 5, text.strip())
        else:
            self._safe_multi_cell(available_width, 5, text.strip())

    def add_paragraph(self, text: str) -> None:
        """Add a regular paragraph."""
        self.set_font(self.font_name, "", self.config.body_size)
        self.set_text_color(*self.config.text_primary)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5, text.strip(), align="L")
        self.ln(2)

    def add_bold_title(self, text: str) -> None:
        """Add a standalone bold title (for project/publication entries without colon)."""
        self.set_x(self.l_margin)
        self.set_font(self.font_name, "B", self.config.body_size)
        if self.style == CVStyle.MODERN:
            self.set_text_color(*self.config.text_primary)
        else:
            self.set_text_color(*self.config.text_primary)
        self.multi_cell(0, 5, text.strip(), align="L")
        self.set_text_color(*self.config.text_primary)
        self.set_font(self.font_name, "", self.config.body_size)

    def add_inline_bold_text(self, bold_part: str, regular_part: str) -> None:
        """Add text with bold part followed by regular part on the same line.

        Format: "**Bold** Regular text" renders as bold title inline with regular text.
        """
        self.set_x(self.l_margin)

        # Write bold part
        self.set_font(self.font_name, "B", self.config.body_size)
        self.set_text_color(*self.config.text_primary)
        bold_width = self.get_string_width(bold_part + " ") + 1
        self.cell(bold_width, 5, bold_part + " ")

        # Write regular part
        self.set_font(self.font_name, "", self.config.body_size)
        self.write(5, regular_part.strip())
        self.ln(6)

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
        page_bottom = self.h - self.b_margin

        for skill in skill_list:
            self.set_font(self.font_name, "", 8)
            pill_width = self.get_string_width(skill) + 6

            # Check if we need to wrap to next line
            if x + pill_width > max_x:
                x = self.l_margin
                y += row_height

            # Check if we need a page break
            if y + row_height > page_bottom:
                self.add_page()
                y = self.get_y()
                x = self.l_margin

            # Draw the pill
            self._draw_skill_pill(skill, x, y)
            x += pill_width + pill_spacing

        # Move cursor below all pills
        self.set_y(y + row_height + 2)
        self.set_text_color(*self.config.text_primary)
