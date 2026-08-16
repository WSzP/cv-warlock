"""Inline company logos for CV experience headers.

Two ways a logo reaches the PDF, both resolved here into a plain list of
runs that the generator renders left to right:

1. A markdown image token written into the job title, e.g.
   ``### Principal AI Architect at SAP SE ![](sap.svg) | Budapest``
2. A company name the registry knows, needing no markdown syntax at all.

The token wins when both apply, so an explicit placement is never
silently overridden. Everything in this module is pure: no PDF state,
no drawing, just paths and text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fpdf.svg import SVGObject  # type: ignore[import-untyped]

# Project root, matching the layout generator.py assumes for fonts/.
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

LOGO_DIR = _PROJECT_ROOT / "assets" / "logos"

# Company name (lowercase) -> filename in LOGO_DIR. Longest name wins,
# so "SAP SE" is never truncated to "SAP".
COMPANY_LOGOS: dict[str, str] = {
    "sap se": "sap.svg",
    "sap": "sap.svg",
}

# Where a logo's lettering sits, as a fraction of the logo height measured
# from its top. Logos are padded below the wordmark, so aligning the outer
# box would float the lettering above the line of text. 1.0 means the ink
# runs to the bottom edge. To measure a new logo, render it large and divide
# the wordmark's bottom edge by the full logo height.
LOGO_BASELINES: dict[str, float] = {
    "sap.svg": 0.8165,
}

_TOKEN_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

# Only vector logos: the aspect ratio comes from the SVG viewBox.
_ALLOWED_SUFFIXES = {".svg"}


@dataclass(frozen=True)
class TextRun:
    """A stretch of title text, bold before the separator and regular after."""

    text: str
    bold: bool


@dataclass(frozen=True)
class LogoRun:
    """A logo drawn inline between two text runs."""

    path: Path


TitleRun = TextRun | LogoRun


def resolve_logo_path(raw: str) -> Path | None:
    """Resolve a logo reference to a file inside the project, or None.

    Tries, in order: bare filename against LOGO_DIR, the path against the
    project root, then the path as given. Markdown pasted into the
    md_to_pdf tool has no file to anchor relative paths to, so the first
    two forms are what make `![](sap.svg)` and `![](assets/logos/sap.svg)`
    both work regardless of the process working directory.

    Anything resolving outside the project is rejected: pasted markdown
    must not be able to point the renderer at arbitrary files.
    """
    candidate = Path(raw)
    if candidate.suffix.lower() not in _ALLOWED_SUFFIXES:
        return None
    if candidate.is_absolute():
        return None

    for attempt in (LOGO_DIR / candidate.name, _PROJECT_ROOT / candidate, candidate):
        try:
            resolved = attempt.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        if not resolved.is_relative_to(_PROJECT_ROOT):
            continue
        return resolved

    return None


def extract_logo_token(text: str) -> tuple[str, Path | None, int]:
    """Pull the first markdown image token out of a line.

    Returns the text without the token, the resolved logo path (None if the
    file is missing, so a typo shows up as a missing logo rather than a
    printed token), and the index into the cleaned text where the logo
    belongs. The offset is -1 when there is no token.
    """
    match = _TOKEN_RE.search(text)
    if not match:
        return text, None, -1

    prefix = text[: match.start()].rstrip()
    suffix = text[match.end() :]
    return prefix + suffix, resolve_logo_path(match.group(1)), len(prefix)


def find_company_logo(text: str) -> tuple[Path, int] | None:
    """Find a registered company name in the text and where its logo goes.

    Returns the logo path and the index just past the company name, or None
    when no known company is mentioned. Matching is case-insensitive and
    respects word boundaries, so "SAPPHIRE" is not a SAP mention.
    """
    for name in sorted(COMPANY_LOGOS, key=len, reverse=True):
        match = re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE)
        if not match:
            continue
        path = resolve_logo_path(COMPANY_LOGOS[name])
        if path is None:
            continue
        return path, match.end()

    return None


def _split_on_separator(text: str) -> list[TextRun]:
    """Split a title into its bold lead and regular remainder.

    Mirrors the existing header rule: everything before the first pipe or
    en-dash is bold, the separator and everything after it is not.
    """
    separator_at = -1
    for separator in ("|", "–"):
        found = text.find(separator)
        if found != -1:
            separator_at = found
            break

    if separator_at == -1:
        return [TextRun(text, bold=True)]

    return [
        TextRun(text[:separator_at].rstrip(), bold=True),
        TextRun(" " + text[separator_at:].strip(), bold=False),
    ]


def plan_title_runs(title: str) -> list[TitleRun]:
    """Plan an experience title into ordered text and logo runs.

    An explicit markdown token places the logo where it was written; failing
    that, a known company name places one just after the name.
    """
    text, token_path, token_at = extract_logo_token(title)

    if token_at >= 0:
        logo = None if token_path is None else (token_path, token_at)
    else:
        logo = find_company_logo(text)

    text_runs = _split_on_separator(text)
    if logo is None:
        return [run for run in text_runs if run.text]

    logo_path, logo_at = logo
    runs: list[TitleRun] = []
    cursor = 0
    placed = False

    for run in text_runs:
        end = cursor + len(run.text)
        if not placed and logo_at <= end:
            head = run.text[: logo_at - cursor]
            tail = run.text[logo_at - cursor :]
            if head:
                runs.append(TextRun(head, run.bold))
            runs.append(LogoRun(logo_path))
            if tail:
                runs.append(TextRun(tail, run.bold))
            placed = True
        elif run.text:
            runs.append(run)
        cursor = end

    if not placed:
        runs.append(LogoRun(logo_path))

    return runs


def logo_baseline_ratio(path: Path) -> float:
    """How far down the logo its lettering sits, as a fraction of its height.

    Defaults to 1.0, which rests the logo's bottom edge on the text baseline.
    """
    return LOGO_BASELINES.get(path.name.lower(), 1.0)


def logo_aspect_ratio(path: Path) -> float:
    """Width-to-height ratio of an SVG, taken from its viewBox.

    Read through fpdf2's own SVG parser so the ratio matches exactly what
    it will draw when given a height alone.
    """
    svg = SVGObject.from_file(str(path))
    viewbox = svg.viewbox
    if viewbox and viewbox[3]:
        return float(viewbox[2]) / float(viewbox[3])

    width, height = svg.width, svg.height
    if isinstance(width, int | float) and isinstance(height, int | float) and height:
        return float(width) / float(height)

    raise ValueError(f"No viewBox or usable dimensions in {path}")
