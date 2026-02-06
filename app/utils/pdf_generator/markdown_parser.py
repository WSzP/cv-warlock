"""Markdown CV parser and text sanitization utilities.

Parses markdown-formatted CVs into structured sections, and provides
text sanitization for malformed markdown and unsupported characters.
"""

import re
from typing import Any


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
