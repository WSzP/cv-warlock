"""PDF text extraction utility with LinkedIn-specific cleaning."""

import os
import re
from typing import BinaryIO

# Default cutoff year for experience - experiences ending before this year are excluded
DEFAULT_EXPERIENCE_CUTOFF_YEAR = int(os.getenv("CV_WARLOCK_EXPERIENCE_CUTOFF_YEAR", "2012"))

# Romanian diacritic fixes: LinkedIn uses cedilla (ş ţ) instead of comma below (ș ț)
ROMANIAN_DIACRITIC_MAP = {
    "ş": "ș",  # s with cedilla -> s with comma below
    "ţ": "ț",  # t with cedilla -> t with comma below
    "Ş": "Ș",  # S with cedilla -> S with comma below
    "Ţ": "Ț",  # T with cedilla -> T with comma below
}


def fix_romanian_diacritics(text: str) -> str:
    """Fix Romanian diacritics from LinkedIn's incorrect cedilla to proper comma-below.

    LinkedIn exports use the Turkish-style cedilla characters (ş, ţ) instead of
    the correct Romanian comma-below characters (ș, ț).
    """
    for wrong, correct in ROMANIAN_DIACRITIC_MAP.items():
        text = text.replace(wrong, correct)
    return text


def extract_text_from_pdf(
    file: BinaryIO,
    fix_romanian: bool = True,
    experience_cutoff_year: int | None = None,
) -> tuple[str | None, str | None]:
    """Extract text content from a PDF file.

    Args:
        file: File-like object containing PDF data.
        fix_romanian: Whether to fix Romanian diacritics (LinkedIn uses wrong characters).
        experience_cutoff_year: Exclude experiences ending before this year.
            If None, uses CV_WARLOCK_EXPERIENCE_CUTOFF_YEAR env var or 2012 default.

    Returns:
        Tuple of (extracted_text, error_message). One will be None.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None, "PDF parsing library not installed. Run: uv add pymupdf"

    # Resolve cutoff year
    if experience_cutoff_year is None:
        experience_cutoff_year = DEFAULT_EXPERIENCE_CUTOFF_YEAR

    try:
        # Read PDF content
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Extract text from all pages
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(text)

        doc.close()

        if not text_parts:
            return None, "No text content found in PDF"

        # Join pages with double newlines
        full_text = "\n\n".join(text_parts)

        # Fix Romanian diacritics if requested (LinkedIn uses cedilla instead of comma-below)
        if fix_romanian:
            full_text = fix_romanian_diacritics(full_text)

        # Detect if this is a LinkedIn PDF and apply appropriate cleaning
        if _is_linkedin_pdf(full_text):
            full_text = clean_linkedin_pdf(full_text, experience_cutoff_year)
        else:
            full_text = clean_extracted_text(full_text)

        if len(full_text) < 50:
            return None, "Extracted text is too short. The PDF may be image-based."

        return full_text, None

    except Exception as e:
        return None, f"Error reading PDF: {str(e)}"


def _is_linkedin_pdf(text: str) -> bool:
    """Detect if the PDF is a LinkedIn export."""
    linkedin_indicators = [
        "linkedin.com/in/",
        "www.linkedin.com",
        "Top Skills",
        "Page 1 of",
        "(LinkedIn)",
    ]
    text_lower = text.lower()
    matches = sum(1 for indicator in linkedin_indicators if indicator.lower() in text_lower)
    return matches >= 2


def clean_linkedin_pdf(text: str, experience_cutoff_year: int = 2012) -> str:
    """Clean LinkedIn PDF export into a proper CV format.

    LinkedIn PDFs have specific formatting issues:
    - "Page X of Y" markers throughout
    - Sidebar content (Contact, Skills, Languages) mixed with main content
    - Awkward line breaks from PDF layout
    - Section headers that need restructuring

    Args:
        text: Raw PDF text content.
        experience_cutoff_year: Exclude experiences ending before this year.
    """
    # Remove page markers (Page 1 of 6, etc.)
    text = re.sub(r"Page \d+ of \d+\n?", "", text)

    # Extract and restructure sections
    sections = _parse_linkedin_sections(text)

    # Filter experiences by cutoff year
    if sections["experience"]:
        sections["experience"] = _filter_experiences_by_cutoff(
            sections["experience"], experience_cutoff_year
        )

    # Build clean CV format
    return _build_cv_from_sections(sections)


def _filter_experiences_by_cutoff(experiences: list, cutoff_year: int) -> list:
    """Filter out experiences that ended before the cutoff year.

    Args:
        experiences: List of experience dictionaries with 'dates' field.
        cutoff_year: Exclude experiences ending before this year.

    Returns:
        Filtered list of experiences.
    """
    filtered = []
    for exp in experiences:
        dates = exp.get("dates", "")

        # If still present, always include
        if "Present" in dates or "present" in dates:
            filtered.append(exp)
            continue

        # Extract end year from date range
        # Patterns: "October 2022 - August 2025", "2007 - 2008", "September 2015 - May 2017"
        end_year_match = re.search(r"[-–]\s*(?:\w+\s+)?(\d{4})", dates)
        if end_year_match:
            end_year = int(end_year_match.group(1))
            if end_year >= cutoff_year:
                filtered.append(exp)
        else:
            # If we can't parse, include it to be safe
            filtered.append(exp)

    return filtered


def _parse_linkedin_sections(text: str) -> dict:
    """Parse LinkedIn PDF into logical sections."""
    sections = {
        "name": "",
        "headline": "",
        "location": "",
        "contact": {},
        "summary": "",
        "experience": [],
        "education": [],
        "skills": [],
        "languages": [],
        "certifications": [],
        "publications": [],
    }

    lines = text.split("\n")
    current_section = None
    current_content = []

    # LinkedIn section headers (order matters for proper parsing)
    section_markers = {
        "Contact": "contact_raw",
        "Top Skills": "skills",
        "Languages": "languages",
        "Certifications": "certifications",
        "Publications": "publications",
        "Summary": "summary",
        "Experience": "experience",
        "Education": "education",
    }

    # First pass: find the name/headline which appears after Publications in LinkedIn PDF
    # LinkedIn PDF layout: Contact, Skills, Languages, Certs, Publications, [NAME], [HEADLINE], [LOCATION], Summary...
    found_publications = False
    publication_lines = []

    for i, line in enumerate(lines):
        line = line.strip()
        if line == "Publications":
            found_publications = True
            continue
        if found_publications and line == "Summary":
            break
        if found_publications:
            publication_lines.append((i, line))

    # The name is usually the first line after publication titles that looks like a person name
    # Person names: 2-4 words, mostly letters, no special keywords
    # Note: We check if line STARTS with these keywords (not just contains) to avoid false positives
    publication_title_starts = [
        "the ",
        "a ",
    ]
    # These keywords indicate publication titles when they appear anywhere
    publication_keywords_anywhere = [
        "book",
        "article",
        "paper",
        "journal",
        "volume",
        "edition",
    ]

    def is_publication_title(text: str) -> bool:
        """Check if this line looks like a publication title."""
        text_lower = text.lower()
        # Check for keywords that indicate publication when at start
        if any(text_lower.startswith(kw) for kw in publication_title_starts):
            return True
        # Check for keywords that indicate publication anywhere
        if any(kw in text_lower for kw in publication_keywords_anywhere):
            return True
        return False

    def looks_like_name(text: str) -> bool:
        """Check if text looks like a person's name."""
        words = text.split()
        if not (2 <= len(words) <= 4):
            return False
        # All words should start with uppercase and be mostly letters
        if not all(w[0].isupper() for w in words if w):
            return False
        if not all(c.isalpha() or c in ". -" for c in text):
            return False
        # Exclude lines that contain publication/title-like words
        title_words = [
            "experience",
            "mapping",
            "book",
            "guide",
            "learning",
            "science",
            "design",
            "development",
            "business",
        ]
        text_lower = text.lower()
        if any(tw in text_lower for tw in title_words):
            return False
        return True

    headline_lines = []
    for _, (i, line) in enumerate(publication_lines):
        if not line:
            continue

        # First check if this looks like a person's name (before checking publications)
        if looks_like_name(line) and not sections["name"]:
            sections["name"] = line
            continue

        # Skip obvious publication titles (but only if we haven't found the name yet)
        # After finding the name, we're in headline territory
        if not sections["name"] and is_publication_title(line):
            continue

        # Check if this looks like a location line (ends headline collection)
        is_location = (
            "," in line
            and len(line) < 60
            and "★" not in line
            and not line.startswith("&")
            and not any(
                kw in line.lower()
                for kw in ["head", "founder", "manager", "director", "ceo", "author"]
            )
        )

        if sections["name"] and is_location and not sections["location"]:
            # This is the location - finalize headline first
            if headline_lines:
                sections["headline"] = " ".join(headline_lines)
            sections["location"] = line
            break

        if sections["name"] and not sections["location"]:
            # Headline often has stars, dashes, job titles, or company names
            # Or continues from previous headline line (starts with & or lowercase)
            if (
                "★" in line
                or " - " in line
                or line.startswith("&")
                or (headline_lines and line[0].islower())
                or any(
                    kw in line.lower()
                    for kw in [
                        "head",
                        "founder",
                        "manager",
                        "director",
                        "ceo",
                        "cto",
                        "researcher",
                        "author",
                        "curator",
                        "past:",
                        "genai",
                        "autonomous",
                        "swarms",
                        "orchestrators",
                    ]
                )
            ):
                headline_lines.append(line)
                continue

    # Finalize headline if we haven't yet (no location found)
    if headline_lines and not sections["headline"]:
        sections["headline"] = " ".join(headline_lines)

    # Second pass: parse sections
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Check for section markers
        section_found = None
        for marker, section_key in section_markers.items():
            if line == marker:
                # Save previous section content
                if current_section and current_content:
                    _save_section_content(sections, current_section, current_content)
                current_section = section_key
                current_content = []
                section_found = True
                break

        if section_found:
            i += 1
            continue

        # For publications section, stop collecting when we hit the name
        if current_section == "publications" and line == sections["name"]:
            _save_section_content(sections, current_section, current_content)
            current_section = "name_block"  # Skip name/headline/location block
            current_content = []
            i += 1
            continue

        # Skip name block content (already extracted)
        if current_section == "name_block":
            if line == "Summary":
                current_section = "summary"
                current_content = []
            i += 1
            continue

        # Collect content for current section
        if current_section and current_section not in ["name_block"]:
            current_content.append(line)

        i += 1

    # Save last section
    if current_section and current_content and current_section not in ["name_block"]:
        _save_section_content(sections, current_section, current_content)

    return sections


def _save_section_content(sections: dict, section_key: str, content: list):
    """Process and save section content."""
    if section_key == "contact_raw":
        sections["contact"] = _parse_contact(content)
    elif section_key == "skills":
        sections["skills"] = _parse_list_section(content)
    elif section_key == "languages":
        sections["languages"] = _parse_languages(content)
    elif section_key == "certifications":
        sections["certifications"] = _parse_list_section(content)
    elif section_key == "publications":
        sections["publications"] = _parse_list_section(content)
    elif section_key == "summary":
        sections["summary"] = _clean_text_block("\n".join(content))
    elif section_key == "experience":
        sections["experience"] = _parse_experience(content)
    elif section_key == "education":
        sections["education"] = _parse_education(content)


def _parse_contact(lines: list) -> dict:
    """Parse contact information."""
    contact = {}
    for line in lines:
        line = line.strip()
        if "(Mobile)" in line or "(Phone)" in line:
            contact["phone"] = re.sub(r"\s*\(Mobile\)|\(Phone\)", "", line).strip()
        elif "@" in line and "." in line:
            contact["email"] = line
        elif "linkedin.com" in line.lower():
            contact["linkedin"] = line
        elif line.startswith("www.") or line.startswith("http"):
            contact["website"] = line
    return contact


def _parse_list_section(lines: list) -> list:
    """Parse a simple list section (skills, certifications, etc.)."""
    items = []
    current_item = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip LinkedIn-specific labels
        if line in ["(LinkedIn)", "(Mobile)", "(Phone)"]:
            continue

        # Check if this line looks like it continues the previous one
        # Signs of continuation:
        # - Starts with lowercase letter
        # - Previous line ends with incomplete phrase (preposition, colon)
        # - Single capitalized word that looks like a continuation (e.g., "Python", "Networks")
        # - Line that doesn't look like a complete item title
        is_continuation = False
        if current_item:
            prev_text = current_item[-1]
            # Lowercase start = continuation
            if line[0].islower():
                is_continuation = True
            # Previous line ends with preposition/connector
            elif prev_text.endswith(("and", "in", "of", "for", "with", ":")):
                is_continuation = True
            # Previous line ends with incomplete certification title pattern
            elif re.search(
                r"(Neural|Convolutional|Deep|Machine|Natural|Natural Language|Processing|Learning)$",
                prev_text,
            ):
                is_continuation = True
            # Current line looks like a continuation (single word or "in Python" etc.)
            elif re.match(
                r"^(in\s|for\s|with\s|Networks|Python|JavaScript|Java\b|Processing)", line
            ):
                is_continuation = True

        if is_continuation:
            current_item.append(line)
        else:
            # New item
            if current_item:
                items.append(" ".join(current_item))
            current_item = [line]

    if current_item:
        items.append(" ".join(current_item))

    return items


def _parse_languages(lines: list) -> list:
    """Parse languages with proficiency levels."""
    languages = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Language format: "German (Elementary)" or just "German"
        if "(" in line:
            languages.append(line)
        elif line and not line.startswith("Page"):
            languages.append(line)
    return languages


def _parse_experience(lines: list) -> list:
    """Parse experience section into structured entries."""
    experiences = []
    current_exp = None
    current_description = []

    # Title keywords for detecting job titles (must appear near start of line)
    title_keywords = r"^(Director|Manager|Head of|Head,|Founder|CEO|CTO|CFO|COO|Engineer|Developer|Designer|Consultant|Lead\b|Senior|Junior|Curator|Expert|Specialist|Analyst|Architect|Administrator|Coordinator|VP\b|President|Chief|Officer|User Experience|UX |UI |Conference Curator|Blockchain|Webdesigner)"

    # Words that indicate a line is description, not a title
    description_indicators = [
        "responsible for",
        "since",
        " by ",
        " of the ",
        " I ",
        " we ",
        "helped",
        "including",
        "such as",
        "delivering",
        "working",
        "supporting",
        "leading",
        "managing",
        "developing",
    ]

    def is_job_title(text: str) -> bool:
        """Check if text looks like a job title."""
        # Job titles are usually short (< 60 chars) and don't contain description words
        if len(text) > 60:
            return False
        # Must not contain description indicators
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in description_indicators):
            return False
        return bool(re.search(title_keywords, text, re.I))

    def is_date_line(text: str) -> bool:
        """Check if text is a date range."""
        return bool(
            re.match(
                r"^(January|February|March|April|May|June|July|August|September|October|November|December|\d{4})",
                text,
            )
        )

    def is_location_line(text: str) -> bool:
        """Check if text looks like a location (City, Region, Country)."""
        # Must have comma, be relatively short, and not contain job-related words
        if "," not in text or len(text) > 60:
            return False
        # Location lines typically don't start with verbs or have long descriptions
        non_location_starts = [
            "curator",
            "led",
            "built",
            "managed",
            "developed",
            "created",
            "and ",
            "strong",
        ]
        if any(text.lower().startswith(x) for x in non_location_starts):
            return False
        # Should look like "City, State/Country"
        return bool(re.match(r"^[A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s,]+$", text))

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Check if this line is a company name (followed by a job title)
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        is_company_line = (
            len(line) > 2
            and not line.startswith("-")
            and not line.startswith("•")
            and not is_date_line(line)
            and not is_location_line(line)
            and is_job_title(next_line)
        )

        if is_company_line:
            # Save previous experience
            if current_exp:
                current_exp["description"] = _clean_text_block("\n".join(current_description))
                experiences.append(current_exp)

            current_exp = {
                "company": line,
                "title": "",
                "dates": "",
                "location": "",
                "description": "",
            }
            current_description = []
            i += 1
            continue

        # Check if this is a job title (when we have a current experience without title)
        if current_exp and not current_exp["title"] and is_job_title(line):
            current_exp["title"] = line
            i += 1
            continue

        # Check for date line
        if current_exp and not current_exp["dates"] and is_date_line(line):
            current_exp["dates"] = line
            i += 1
            continue

        # Check for location line (only if we have dates already)
        if (
            current_exp
            and current_exp["dates"]
            and not current_exp["location"]
            and is_location_line(line)
        ):
            current_exp["location"] = line
            i += 1
            continue

        # Check if this might be a new company (lookahead shows it's followed by title pattern)
        # This handles cases where we're in description but hit a new company
        if current_exp and current_exp["title"]:
            # Check if current line could be a company name for next entry
            if i + 1 < len(lines):
                peek_next = lines[i + 1].strip()
                if (
                    is_job_title(peek_next)
                    and not line.startswith("-")
                    and not line.startswith("•")
                    and not is_date_line(line)
                    and not is_location_line(line)
                ):
                    # This looks like a new company - save current and start new
                    current_exp["description"] = _clean_text_block("\n".join(current_description))
                    experiences.append(current_exp)

                    current_exp = {
                        "company": line,
                        "title": "",
                        "dates": "",
                        "location": "",
                        "description": "",
                    }
                    current_description = []
                    i += 1
                    continue

        # If we have a current experience, add to description
        if current_exp:
            current_description.append(line)
        else:
            # No current experience - try to start one if this looks like a title
            if is_job_title(line):
                # Look backward for company name
                company = lines[i - 1].strip() if i > 0 else ""
                current_exp = {
                    "company": company,
                    "title": line,
                    "dates": "",
                    "location": "",
                    "description": "",
                }
                current_description = []

        i += 1

    # Save last experience
    if current_exp:
        current_exp["description"] = _clean_text_block("\n".join(current_description))
        experiences.append(current_exp)

    return experiences


def _parse_education(lines: list) -> list:
    """Parse education section."""
    education = []
    current_edu = None
    institution_lines = []

    # Institution markers - lines containing these start a new education entry
    institution_markers = ["University", "College", "Liceul", "Universitatea", "Institut"]

    def is_institution_start(text: str) -> bool:
        """Check if line starts a new institution."""
        return any(marker in text for marker in institution_markers)

    def is_degree_line(text: str) -> bool:
        """Check if line contains degree information."""
        degree_keywords = [
            "Bachelor",
            "Master",
            "PhD",
            "Doctorate",
            "degree",
            "High School",
            "Research",
        ]
        return any(kw in text for kw in degree_keywords)

    def finalize_education(edu: dict, inst_lines: list) -> dict:
        """Finalize an education entry by joining institution lines."""
        if inst_lines:
            edu["institution"] = " ".join(inst_lines)
        # Clean up dates - remove incomplete dates that are just month names
        if edu["dates"]:
            # If date is just a month or incomplete, try to find the full date
            dates = edu["dates"]
            # Remove leading/trailing parentheses and whitespace
            dates = dates.strip("() ")
            # Check if it's just a month name without year
            if dates in [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]:
                # Incomplete date, clear it
                edu["dates"] = ""
            else:
                edu["dates"] = dates
        return edu

    # First pass: join lines that are continuations of previous lines
    # This handles dates split across lines like "October\n2018 - 2021"
    joined_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if this line looks like a date continuation (starts with year)
        if joined_lines and re.match(r"^\d{4}\s*[-–]", line):
            # This is likely a continuation of the previous date
            joined_lines[-1] = joined_lines[-1] + " " + line
        # Check if previous line ended with month name and this is a date
        elif joined_lines and re.search(
            r"\((January|February|March|April|May|June|July|August|September|October|November|December)$",
            joined_lines[-1],
        ):
            joined_lines[-1] = joined_lines[-1] + " " + line
        # Check if this line is just a closing parenthesis and year
        elif joined_lines and re.match(r"^\d{4}\)?$", line):
            joined_lines[-1] = joined_lines[-1] + " " + line
        # Check if previous line ends with open parenthesis or mid-word
        # (handles cases like "Design\nIndustrial)" where text is split)
        elif joined_lines and (
            joined_lines[-1].endswith("(")
            or (
                re.search(r"[a-zA-Z]$", joined_lines[-1])
                and re.match(r"^[a-zA-Z)]", line)
                and not is_institution_start(line)
                and not is_degree_line(line)  # Don't join if this is a new degree line
            )
        ):
            # Add space if not ending with ( and line doesn't start with )
            separator = "" if joined_lines[-1].endswith("(") or line.startswith(")") else " "
            joined_lines[-1] = joined_lines[-1] + separator + line
        else:
            joined_lines.append(line)

    i = 0
    while i < len(joined_lines):
        line = joined_lines[i]
        if not line:
            i += 1
            continue

        # Check if this starts a new institution
        if is_institution_start(line):
            # Save previous education entry
            if current_edu:
                education.append(finalize_education(current_edu, institution_lines))

            current_edu = {
                "institution": "",
                "degree": "",
                "dates": "",
                "details": "",
            }
            institution_lines = [line]
            i += 1
            continue

        if current_edu:
            # Check if this is a degree line (contains degree info and often dates)
            if is_degree_line(line):
                # Degree lines often have format: "Master's degree, Field · (dates)"
                # Split on · if present
                if "·" in line:
                    parts = line.split("·", 1)
                    current_edu["degree"] = parts[0].strip()
                    if len(parts) > 1:
                        # Extract dates from the second part
                        date_part = parts[1].strip()
                        # Remove parentheses
                        date_part = date_part.strip("() ")
                        current_edu["dates"] = date_part
                else:
                    current_edu["degree"] = line
                i += 1
                continue

            # Check if this looks like a continuation of the institution name
            # (lines that don't start with degree keywords and aren't dates)
            if (
                not is_degree_line(line)
                and not re.search(r"^\(\d{4}", line)
                and not current_edu["degree"]
            ):
                # This might be a continuation of the institution name
                # or additional details after degree
                if not current_edu["degree"]:
                    institution_lines.append(line)
                else:
                    # It's additional details
                    if current_edu["details"]:
                        current_edu["details"] += " " + line
                    else:
                        current_edu["details"] = line
                i += 1
                continue

            # Check for standalone date line
            if re.search(r"\d{4}\s*[-–]\s*\d{4}|\d{4}\s*[-–]\s*Present|\(\w+\s+\d{4}", line):
                if not current_edu["dates"]:
                    current_edu["dates"] = line.strip("() ")
                i += 1
                continue

        i += 1

    # Save last education entry
    if current_edu:
        education.append(finalize_education(current_edu, institution_lines))

    return education


def _clean_text_block(text: str) -> str:
    """Clean a block of text, joining broken lines intelligently."""
    # Remove page markers that might have been missed
    text = re.sub(r"Page \d+ of \d+", "", text)

    lines = text.split("\n")
    cleaned_lines = []
    current_paragraph = []

    for line in lines:
        line = line.strip()
        if not line:
            if current_paragraph:
                cleaned_lines.append(" ".join(current_paragraph))
                current_paragraph = []
            continue

        # Check if this is a bullet point or new paragraph
        if line.startswith("-") or line.startswith("•") or line.startswith("★"):
            if current_paragraph:
                cleaned_lines.append(" ".join(current_paragraph))
                current_paragraph = []
            cleaned_lines.append(line)
        # Check if previous line ended mid-sentence (lowercase continuation)
        elif current_paragraph and line[0].islower():
            current_paragraph.append(line)
        # Check if this looks like a continuation (doesn't start with capital after period)
        elif current_paragraph and not current_paragraph[-1].endswith((".", "!", "?", ":")):
            current_paragraph.append(line)
        else:
            if current_paragraph:
                cleaned_lines.append(" ".join(current_paragraph))
            current_paragraph = [line]

    if current_paragraph:
        cleaned_lines.append(" ".join(current_paragraph))

    return "\n".join(cleaned_lines)


def _build_cv_from_sections(sections: dict) -> str:
    """Build a clean CV text from parsed sections."""
    parts = []

    # Header
    if sections["name"]:
        parts.append(f"# {sections['name']}")
    if sections["headline"]:
        parts.append(sections["headline"])
    if sections["location"]:
        parts.append(sections["location"])

    # Contact
    if sections["contact"]:
        contact_parts = []
        if sections["contact"].get("email"):
            contact_parts.append(sections["contact"]["email"])
        if sections["contact"].get("phone"):
            contact_parts.append(sections["contact"]["phone"])
        if sections["contact"].get("linkedin"):
            contact_parts.append(sections["contact"]["linkedin"])
        if sections["contact"].get("website"):
            contact_parts.append(sections["contact"]["website"])
        if contact_parts:
            parts.append("\n" + " | ".join(contact_parts))

    # Summary
    if sections["summary"]:
        parts.append("\n## Summary\n" + sections["summary"])

    # Skills
    if sections["skills"]:
        parts.append("\n## Skills\n" + ", ".join(sections["skills"]))

    # Experience
    if sections["experience"]:
        parts.append("\n## Experience")
        for exp in sections["experience"]:
            exp_text = f"\n### {exp['title']}"
            if exp["company"]:
                exp_text += f" at {exp['company']}"
            if exp["dates"]:
                exp_text += f"\n{exp['dates']}"
            if exp["location"]:
                exp_text += f" | {exp['location']}"
            if exp["description"]:
                exp_text += f"\n{exp['description']}"
            parts.append(exp_text)

    # Education
    if sections["education"]:
        parts.append("\n## Education")
        for edu in sections["education"]:
            edu_text = f"\n### {edu['institution']}\n"
            # Degree and dates on separate lines
            if edu["degree"]:
                edu_text += f"\n{edu['degree']}"
            if edu["dates"]:
                edu_text += f"\n{edu['dates']}"
            if edu["details"]:
                edu_text += f"\n{edu['details']}"
            parts.append(edu_text)

    # Languages
    if sections["languages"]:
        parts.append("\n## Languages\n" + ", ".join(sections["languages"]))

    # Certifications
    if sections["certifications"]:
        parts.append(
            "\n## Certifications\n" + "\n".join(f"- {cert}" for cert in sections["certifications"])
        )

    # Publications
    if sections["publications"]:
        parts.append(
            "\n## Publications\n" + "\n".join(f"- {pub}" for pub in sections["publications"])
        )

    return "\n".join(parts)


def clean_extracted_text(text: str) -> str:
    """Clean up extracted PDF text (generic, non-LinkedIn).

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    # Remove page headers/footers that are just page numbers
    text = re.sub(r"\n\d+\n", "\n", text)
    text = re.sub(r"Page \d+ of \d+\n?", "", text)

    # Clean up lines
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        # Skip very short lines that are likely artifacts
        if len(line) < 2 and not line.isalnum():
            continue
        lines.append(line)

    return "\n".join(lines).strip()
