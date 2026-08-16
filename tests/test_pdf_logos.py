"""Tests for inline company logos in CV PDF experience headers."""

from pathlib import Path

import fitz  # type: ignore[import-untyped]
import pytest

from app.utils.pdf_generator import generate_cv_pdf
from app.utils.pdf_generator.logos import (
    LOGO_DIR,
    LogoRun,
    TextRun,
    extract_logo_token,
    find_company_logo,
    logo_aspect_ratio,
    plan_title_runs,
    resolve_logo_path,
)

PROJECT_ROOT = Path(__file__).parent.parent
SAP_LOGO = PROJECT_ROOT / "assets" / "logos" / "sap.svg"

# SAP brand blue, roughly #00b1eb -> #0069b4 across the logo gradient.
_SAP_BLUE_MIN = (0, 80, 140)
_SAP_BLUE_MAX = (90, 200, 250)


def _cv_markdown(title: str) -> str:
    """Build a minimal CV around a single experience entry title."""
    return f"""# Peter W. Szabo

+40 754 94 74 74

## Experience

### {title}

March 2026 - Present

- Shipped a governed agentic system on live HANA data.
"""


def _blue_pixel_count(pdf_bytes: bytes) -> int:
    """Count SAP-blue pixels across all pages of a rendered PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = 0
    for page in doc:
        pix = page.get_pixmap(dpi=110)
        samples = pix.samples
        stride = pix.n
        for i in range(0, len(samples), stride):
            r, g, b = samples[i], samples[i + 1], samples[i + 2]
            if (
                _SAP_BLUE_MIN[0] <= r <= _SAP_BLUE_MAX[0]
                and _SAP_BLUE_MIN[1] <= g <= _SAP_BLUE_MAX[1]
                and _SAP_BLUE_MIN[2] <= b <= _SAP_BLUE_MAX[2]
                and b > r + 60
            ):
                total += 1
    return total


def _measure_title_ink(pdf_bytes: bytes, dpi: int = 300) -> dict[str, float]:
    """Measure the rendered title line in millimetres, from actual pixels.

    Returns the logo's size and bottom edge, the text baseline, and the gap
    between the end of the text and the start of the logo.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        hits = page.search_for("Principal AI Architect at SAP SE")
        if not hits:
            continue
        box = hits[0]
        pix = page.get_pixmap(dpi=dpi, clip=fitz.Rect(box.x0 - 2, box.y0 - 2, box.x1 + 40, box.y1))
        per_mm = dpi / 25.4
        blue: list[tuple[int, int]] = []
        dark: list[tuple[int, int]] = []
        light: list[tuple[int, int]] = []
        for y in range(pix.height):
            for x in range(pix.width):
                i = (y * pix.width + x) * pix.n
                r, g, b = pix.samples[i], pix.samples[i + 1], pix.samples[i + 2]
                if b > r + 60 and b > 140 and r < 100:
                    blue.append((x, y))
                elif r < 90 and g < 90 and b < 90:
                    dark.append((x, y))
                elif r > 200 and g > 200 and b > 200:
                    light.append((x, y))
        if not blue:
            raise AssertionError("no logo ink found on the title line")

        logo_left = min(p[0] for p in blue)
        text = [p for p in dark if p[0] < logo_left]

        # The wordmark: light pixels enclosed by logo ink in their own column,
        # which excludes the page showing through the logo's cut-off corner.
        blue_columns: dict[int, list[int]] = {}
        for x, y in blue:
            blue_columns.setdefault(x, []).append(y)
        wordmark = [
            (x, y)
            for x, y in light
            if x in blue_columns and min(blue_columns[x]) < y < max(blue_columns[x])
        ]
        columns: dict[int, int] = {}
        for x, y in text:
            columns[x] = max(columns.get(x, 0), y)
        # Most columns end on the baseline; only descenders reach lower.
        baseline = max(set(columns.values()), key=list(columns.values()).count)

        return {
            "gap": (logo_left - max(p[0] for p in text)) / per_mm,
            "width": (max(p[0] for p in blue) - logo_left) / per_mm,
            "height": (max(p[1] for p in blue) - min(p[1] for p in blue)) / per_mm,
            "logo_bottom": max(p[1] for p in blue) / per_mm,
            "wordmark_bottom": max(p[1] for p in wordmark) / per_mm,
            "baseline": baseline / per_mm,
        }

    raise AssertionError("title line not found in the PDF")


def _page_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


class TestLogoAssets:
    """The bundled logo asset and its geometry."""

    def test_sap_logo_asset_exists(self) -> None:
        assert SAP_LOGO.exists()
        assert LOGO_DIR == PROJECT_ROOT / "assets" / "logos"

    def test_aspect_ratio_comes_from_viewbox(self) -> None:
        """412.38 x 204 viewBox -> width is ~2.02x the height."""
        assert logo_aspect_ratio(SAP_LOGO) == pytest.approx(412.38 / 204, rel=1e-4)


class TestLogoPathResolution:
    """Logo paths from pasted markdown resolve safely inside the project."""

    def test_bare_filename_resolves_against_logo_dir(self) -> None:
        assert resolve_logo_path("sap.svg") == SAP_LOGO

    def test_project_relative_path_resolves(self) -> None:
        assert resolve_logo_path("assets/logos/sap.svg") == SAP_LOGO

    def test_dot_dot_relative_path_resolves(self) -> None:
        """The form that also previews correctly from app/data/cvs/my_cv.md."""
        assert resolve_logo_path("../../../assets/logos/sap.svg") == SAP_LOGO

    def test_missing_file_resolves_to_none(self) -> None:
        assert resolve_logo_path("no-such-logo.svg") is None

    def test_path_outside_project_is_rejected(self) -> None:
        """Pasted markdown must not point the renderer at arbitrary files."""
        assert resolve_logo_path("../../../../../../Windows/win.ini") is None
        assert resolve_logo_path("/etc/passwd") is None


class TestLogoToken:
    """Markdown image tokens are extracted, never printed."""

    def test_token_is_stripped_and_resolved(self) -> None:
        """Offset indexes the cleaned text: the logo sits where the token was."""
        text, path, offset = extract_logo_token("at SAP SE ![](sap.svg) | Budapest")
        assert text == "at SAP SE | Budapest"
        assert path == SAP_LOGO
        assert offset == len("at SAP SE")

    def test_token_with_alt_text_is_stripped(self) -> None:
        text, path, offset = extract_logo_token("at SAP SE ![SAP](sap.svg)")
        assert text == "at SAP SE"
        assert path == SAP_LOGO
        assert offset == len("at SAP SE")

    def test_broken_token_is_stripped_but_yields_no_logo(self) -> None:
        text, path, _ = extract_logo_token("at SAP SE ![](typo.svg)")
        assert text == "at SAP SE"
        assert path is None

    def test_line_without_token_is_unchanged(self) -> None:
        text, path, offset = extract_logo_token("at SAP SE | Budapest")
        assert text == "at SAP SE | Budapest"
        assert path is None
        assert offset == -1


class TestCompanyRegistry:
    """Known company names get a logo with no markdown syntax at all."""

    def test_company_name_is_matched_case_insensitively(self) -> None:
        match = find_company_logo("Principal AI Architect at sap se | Budapest")
        assert match is not None
        path, offset = match
        assert path == SAP_LOGO
        assert offset == len("Principal AI Architect at sap se")

    def test_longest_company_name_wins(self) -> None:
        """'SAP SE' must not be truncated to the shorter 'SAP' entry."""
        match = find_company_logo("Principal AI Architect at SAP SE | Budapest")
        assert match is not None
        assert match[1] == len("Principal AI Architect at SAP SE")

    def test_company_name_needs_word_boundaries(self) -> None:
        assert find_company_logo("Speaker at SAPPHIRE NOW | Orlando") is None

    def test_unknown_company_gets_no_logo(self) -> None:
        assert find_company_logo("Founder at Tengrai | Budapest") is None


class TestTitleRunPlanning:
    """Titles are planned into ordered bold/regular text and logo runs."""

    def test_registry_logo_lands_after_company_name(self) -> None:
        runs = plan_title_runs("Principal AI Architect at SAP SE | Budapest, Hungary")
        assert runs == [
            TextRun("Principal AI Architect at SAP SE", bold=True),
            LogoRun(SAP_LOGO),
            TextRun(" | Budapest, Hungary", bold=False),
        ]

    def test_explicit_token_overrides_registry(self) -> None:
        """One logo, placed where the token was, not where the registry would put it."""
        runs = plan_title_runs("Architect ![](sap.svg) at SAP SE | Budapest")
        logos = [r for r in runs if isinstance(r, LogoRun)]
        assert len(logos) == 1
        assert runs[0] == TextRun("Architect", bold=True)
        assert runs[1] == LogoRun(SAP_LOGO)

    def test_title_without_separator_is_all_bold(self) -> None:
        runs = plan_title_runs("Principal AI Architect at SAP SE")
        assert runs == [
            TextRun("Principal AI Architect at SAP SE", bold=True),
            LogoRun(SAP_LOGO),
        ]

    def test_title_without_logo_yields_text_runs_only(self) -> None:
        runs = plan_title_runs("Founder at Tengrai | Budapest")
        assert all(isinstance(r, TextRun) for r in runs)


class TestRenderedPdf:
    """End-to-end through generate_cv_pdf, the path md_to_pdf.py uses."""

    def test_registry_logo_is_drawn(self) -> None:
        with_logo = generate_cv_pdf(_cv_markdown("Principal AI Architect at SAP SE | Budapest"))[0]
        without_logo = generate_cv_pdf(_cv_markdown("Founder at Tengrai | Budapest"))[0]
        assert _blue_pixel_count(with_logo) > _blue_pixel_count(without_logo) + 100

    def test_title_text_survives_logo_rendering(self) -> None:
        """The regression that matters: text must stay text, for ATS parsing."""
        pdf, _ = generate_cv_pdf(_cv_markdown("Principal AI Architect at SAP SE | Budapest"))
        text = _page_text(pdf)
        assert "Principal AI Architect at SAP SE" in text
        assert "Budapest" in text

    def test_markdown_token_never_appears_as_text(self) -> None:
        pdf, _ = generate_cv_pdf(
            _cv_markdown("Principal AI Architect at SAP SE ![](sap.svg) | Budapest")
        )
        text = _page_text(pdf)
        assert "![]" not in text
        assert "sap.svg" not in text
        assert "Principal AI Architect at SAP SE" in text

    def test_token_and_registry_together_draw_one_logo(self) -> None:
        tokened = generate_cv_pdf(
            _cv_markdown("Principal AI Architect at SAP SE ![](sap.svg) | Budapest")
        )[0]
        registry = generate_cv_pdf(_cv_markdown("Principal AI Architect at SAP SE | Budapest"))[0]
        assert _blue_pixel_count(tokened) == pytest.approx(_blue_pixel_count(registry), rel=0.2)

    def test_missing_logo_file_still_renders_the_title(self) -> None:
        pdf, _ = generate_cv_pdf(
            _cv_markdown("Principal AI Architect at SAP SE ![](typo.svg) | Budapest")
        )
        text = _page_text(pdf)
        assert "Principal AI Architect at SAP SE" in text
        assert "typo.svg" not in text

    def test_logo_keeps_the_pdf_small(self) -> None:
        """A vector logo costs kilobytes; outlined text costs megabytes."""
        pdf, _ = generate_cv_pdf(_cv_markdown("Principal AI Architect at SAP SE | Budapest"))
        assert len(pdf) < 60_000

    def test_logo_wordmark_sits_on_the_text_baseline(self) -> None:
        """The SAP inside the logo must line up with the SAP SE beside it.

        The blue block is padded below the lettering, so aligning the block
        itself would float the wordmark above the line of text.
        """
        pdf, _ = generate_cv_pdf(_cv_markdown("Principal AI Architect at SAP SE | Budapest"))
        ink = _measure_title_ink(pdf)
        assert ink["wordmark_bottom"] == pytest.approx(ink["baseline"], abs=0.2)

    def test_logo_is_separated_from_the_text_by_about_a_space(self) -> None:
        pdf, _ = generate_cv_pdf(_cv_markdown("Principal AI Architect at SAP SE | Budapest"))
        assert 1.0 < _measure_title_ink(pdf)["gap"] < 2.5

    def test_logo_keeps_the_aspect_ratio_from_the_file(self) -> None:
        pdf, _ = generate_cv_pdf(_cv_markdown("Principal AI Architect at SAP SE | Budapest"))
        ink = _measure_title_ink(pdf)
        assert ink["width"] / ink["height"] == pytest.approx(412.38 / 204, rel=0.02)

    def test_pdf_still_has_embedded_fonts(self) -> None:
        pdf, _ = generate_cv_pdf(_cv_markdown("Principal AI Architect at SAP SE | Budapest"))
        doc = fitz.open(stream=pdf, filetype="pdf")
        assert len(doc[0].get_fonts()) > 0
