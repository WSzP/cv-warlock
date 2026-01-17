"""PDF text extraction utility with LinkedIn-specific cleaning."""

import re
from typing import BinaryIO

# Romanian diacritic fixes: LinkedIn uses cedilla (ş ţ) instead of comma below (ș ț)
ROMANIAN_DIACRITIC_MAP = {
    'ş': 'ș',  # s with cedilla -> s with comma below
    'ţ': 'ț',  # t with cedilla -> t with comma below
    'Ş': 'Ș',  # S with cedilla -> S with comma below
    'Ţ': 'Ț',  # T with cedilla -> T with comma below
}


def fix_romanian_diacritics(text: str) -> str:
    """Fix Romanian diacritics from LinkedIn's incorrect cedilla to proper comma-below.

    LinkedIn exports use the Turkish-style cedilla characters (ş, ţ) instead of
    the correct Romanian comma-below characters (ș, ț).
    """
    for wrong, correct in ROMANIAN_DIACRITIC_MAP.items():
        text = text.replace(wrong, correct)
    return text


def extract_text_from_pdf(file: BinaryIO, fix_romanian: bool = True) -> tuple[str | None, str | None]:
    """Extract text content from a PDF file.

    Args:
        file: File-like object containing PDF data.
        fix_romanian: Whether to fix Romanian diacritics (LinkedIn uses wrong characters).

    Returns:
        Tuple of (extracted_text, error_message). One will be None.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None, "PDF parsing library not installed. Run: uv add pymupdf"

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
            full_text = clean_linkedin_pdf(full_text)
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


def clean_linkedin_pdf(text: str) -> str:
    """Clean LinkedIn PDF export into a proper CV format.

    LinkedIn PDFs have specific formatting issues:
    - "Page X of Y" markers throughout
    - Sidebar content (Contact, Skills, Languages) mixed with main content
    - Awkward line breaks from PDF layout
    - Section headers that need restructuring
    """
    # Remove page markers (Page 1 of 6, etc.)
    text = re.sub(r'Page \d+ of \d+\n?', '', text)

    # Extract and restructure sections
    sections = _parse_linkedin_sections(text)

    # Build clean CV format
    return _build_cv_from_sections(sections)


def _parse_linkedin_sections(text: str) -> dict:
    """Parse LinkedIn PDF into logical sections."""
    sections = {
        'name': '',
        'headline': '',
        'location': '',
        'contact': {},
        'summary': '',
        'experience': [],
        'education': [],
        'skills': [],
        'languages': [],
        'certifications': [],
        'publications': [],
    }

    lines = text.split('\n')
    current_section = None
    current_content = []

    # LinkedIn section headers (order matters for proper parsing)
    section_markers = {
        'Contact': 'contact_raw',
        'Top Skills': 'skills',
        'Languages': 'languages',
        'Certifications': 'certifications',
        'Publications': 'publications',
        'Summary': 'summary',
        'Experience': 'experience',
        'Education': 'education',
    }

    # First pass: find the name/headline which appears after Publications in LinkedIn PDF
    # LinkedIn PDF layout: Contact, Skills, Languages, Certs, Publications, [NAME], [HEADLINE], [LOCATION], Summary...
    found_publications = False
    publication_lines = []

    for i, line in enumerate(lines):
        line = line.strip()
        if line == 'Publications':
            found_publications = True
            continue
        if found_publications and line == 'Summary':
            break
        if found_publications:
            publication_lines.append((i, line))

    # The name is usually the first line after publication titles that looks like a person name
    # Person names: 2-4 words, mostly letters, no special keywords
    publication_keywords = ['book', 'article', 'paper', 'journal', 'volume', 'edition', 'mapping', 'guide', 'the ', 'a ']

    for idx, (i, line) in enumerate(publication_lines):
        if not line:
            continue
        # Skip obvious publication titles
        if any(kw in line.lower() for kw in publication_keywords):
            continue

        words = line.split()
        # Person names typically: 2-4 words, all start with uppercase, mostly letters
        is_name_like = (
            2 <= len(words) <= 4 and
            all(w[0].isupper() for w in words if w) and
            all(c.isalpha() or c in '. -' for c in line)
        )

        if is_name_like and not sections['name']:
            sections['name'] = line
            continue

        if sections['name'] and not sections['headline']:
            # Headline often has stars, dashes, job titles, or company names
            if '★' in line or ' - ' in line or any(kw in line.lower() for kw in ['head', 'founder', 'manager', 'director', 'ceo', 'cto', 'researcher', 'author', 'curator', 'past:']):
                sections['headline'] = line
                continue

        if sections['name'] and sections['headline'] and not sections['location']:
            # Location is typically "City, Region, Country"
            if ',' in line and len(line) < 60 and not ' - ' in line:
                sections['location'] = line
                break

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
        if current_section == 'publications' and line == sections['name']:
            _save_section_content(sections, current_section, current_content)
            current_section = 'name_block'  # Skip name/headline/location block
            current_content = []
            i += 1
            continue

        # Skip name block content (already extracted)
        if current_section == 'name_block':
            if line == 'Summary':
                current_section = 'summary'
                current_content = []
            i += 1
            continue

        # Collect content for current section
        if current_section and current_section not in ['name_block']:
            current_content.append(line)

        i += 1

    # Save last section
    if current_section and current_content and current_section not in ['name_block']:
        _save_section_content(sections, current_section, current_content)

    return sections


def _save_section_content(sections: dict, section_key: str, content: list):
    """Process and save section content."""
    if section_key == 'contact_raw':
        sections['contact'] = _parse_contact(content)
    elif section_key == 'skills':
        sections['skills'] = _parse_list_section(content)
    elif section_key == 'languages':
        sections['languages'] = _parse_languages(content)
    elif section_key == 'certifications':
        sections['certifications'] = _parse_list_section(content)
    elif section_key == 'publications':
        sections['publications'] = _parse_list_section(content)
    elif section_key == 'summary':
        sections['summary'] = _clean_text_block('\n'.join(content))
    elif section_key == 'experience':
        sections['experience'] = _parse_experience(content)
    elif section_key == 'education':
        sections['education'] = _parse_education(content)


def _parse_contact(lines: list) -> dict:
    """Parse contact information."""
    contact = {}
    for line in lines:
        line = line.strip()
        if '(Mobile)' in line or '(Phone)' in line:
            contact['phone'] = re.sub(r'\s*\(Mobile\)|\(Phone\)', '', line).strip()
        elif '@' in line and '.' in line:
            contact['email'] = line
        elif 'linkedin.com' in line.lower():
            contact['linkedin'] = line
        elif line.startswith('www.') or line.startswith('http'):
            contact['website'] = line
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
        if line in ['(LinkedIn)', '(Mobile)', '(Phone)']:
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
            elif prev_text.endswith(('and', 'in', 'of', 'for', 'with', ':')):
                is_continuation = True
            # Previous line ends with incomplete certification title pattern
            elif re.search(r'(Neural|Convolutional|Deep|Machine|Natural|Natural Language|Processing|Learning)$', prev_text):
                is_continuation = True
            # Current line looks like a continuation (single word or "in Python" etc.)
            elif re.match(r'^(in\s|for\s|with\s|Networks|Python|JavaScript|Java\b|Processing)', line):
                is_continuation = True

        if is_continuation:
            current_item.append(line)
        else:
            # New item
            if current_item:
                items.append(' '.join(current_item))
            current_item = [line]

    if current_item:
        items.append(' '.join(current_item))

    return items


def _parse_languages(lines: list) -> list:
    """Parse languages with proficiency levels."""
    languages = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Language format: "German (Elementary)" or just "German"
        if '(' in line:
            languages.append(line)
        elif line and not line.startswith('Page'):
            languages.append(line)
    return languages


def _parse_experience(lines: list) -> list:
    """Parse experience section into structured entries."""
    experiences = []
    current_exp = None
    current_description = []

    # Title keywords for detecting job titles (must appear near start of line)
    title_keywords = r'^(Director|Manager|Head of|Head,|Founder|CEO|CTO|CFO|COO|Engineer|Developer|Designer|Consultant|Lead\b|Senior|Junior|Curator|Expert|Specialist|Analyst|Architect|Administrator|Coordinator|VP\b|President|Chief|Officer|User Experience|UX |UI |Conference Curator|Blockchain)'

    def is_job_title(text: str) -> bool:
        """Check if text looks like a job title."""
        return bool(re.search(title_keywords, text, re.I))

    def is_date_line(text: str) -> bool:
        """Check if text is a date range."""
        return bool(re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December|\d{4})', text))

    def is_location_line(text: str) -> bool:
        """Check if text looks like a location (City, Region, Country)."""
        # Must have comma, be relatively short, and not contain job-related words
        if ',' not in text or len(text) > 60:
            return False
        # Location lines typically don't start with verbs or have long descriptions
        non_location_starts = ['curator', 'led', 'built', 'managed', 'developed', 'created', 'and ', 'strong']
        if any(text.lower().startswith(x) for x in non_location_starts):
            return False
        # Should look like "City, State/Country"
        return bool(re.match(r'^[A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s,]+$', text))

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Check if this line is a company name (followed by a job title)
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
        is_company_line = (
            len(line) > 2 and
            not line.startswith('-') and
            not line.startswith('•') and
            not is_date_line(line) and
            not is_location_line(line) and
            is_job_title(next_line)
        )

        if is_company_line:
            # Save previous experience
            if current_exp:
                current_exp['description'] = _clean_text_block('\n'.join(current_description))
                experiences.append(current_exp)

            current_exp = {
                'company': line,
                'title': '',
                'dates': '',
                'location': '',
                'description': '',
            }
            current_description = []
            i += 1
            continue

        # Check if this is a job title (when we have a current experience without title)
        if current_exp and not current_exp['title'] and is_job_title(line):
            current_exp['title'] = line
            i += 1
            continue

        # Check for date line
        if current_exp and not current_exp['dates'] and is_date_line(line):
            current_exp['dates'] = line
            i += 1
            continue

        # Check for location line (only if we have dates already)
        if current_exp and current_exp['dates'] and not current_exp['location'] and is_location_line(line):
            current_exp['location'] = line
            i += 1
            continue

        # Check if this might be a new company (lookahead shows it's followed by title pattern)
        # This handles cases where we're in description but hit a new company
        if current_exp and current_exp['title']:
            # Check if current line could be a company name for next entry
            if i + 1 < len(lines):
                peek_next = lines[i + 1].strip()
                if (is_job_title(peek_next) and
                    not line.startswith('-') and
                    not line.startswith('•') and
                    not is_date_line(line) and
                    not is_location_line(line)):
                    # This looks like a new company - save current and start new
                    current_exp['description'] = _clean_text_block('\n'.join(current_description))
                    experiences.append(current_exp)

                    current_exp = {
                        'company': line,
                        'title': '',
                        'dates': '',
                        'location': '',
                        'description': '',
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
                company = lines[i-1].strip() if i > 0 else ''
                current_exp = {
                    'company': company,
                    'title': line,
                    'dates': '',
                    'location': '',
                    'description': '',
                }
                current_description = []

        i += 1

    # Save last experience
    if current_exp:
        current_exp['description'] = _clean_text_block('\n'.join(current_description))
        experiences.append(current_exp)

    return experiences


def _parse_education(lines: list) -> list:
    """Parse education section."""
    education = []
    current_edu = None
    current_details = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Education entries typically start with university name
        if 'University' in line or 'College' in line or 'Liceul' in line or 'School' in line:
            if current_edu:
                current_edu['details'] = ' '.join(current_details)
                education.append(current_edu)

            current_edu = {
                'institution': line,
                'degree': '',
                'dates': '',
                'details': '',
            }
            current_details = []
        elif current_edu:
            # Check for degree line
            if any(deg in line for deg in ["Bachelor", "Master", "PhD", "Doctorate", "degree", "High School"]):
                current_edu['degree'] = line
            # Check for date range
            elif re.search(r'\d{4}\s*[-–]\s*\d{4}|\d{4}\s*[-–]\s*Present|\(\w+\s+\d{4}', line):
                current_edu['dates'] = line
            else:
                current_details.append(line)

    if current_edu:
        current_edu['details'] = ' '.join(current_details)
        education.append(current_edu)

    return education


def _clean_text_block(text: str) -> str:
    """Clean a block of text, joining broken lines intelligently."""
    # Remove page markers that might have been missed
    text = re.sub(r'Page \d+ of \d+', '', text)

    lines = text.split('\n')
    cleaned_lines = []
    current_paragraph = []

    for line in lines:
        line = line.strip()
        if not line:
            if current_paragraph:
                cleaned_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            continue

        # Check if this is a bullet point or new paragraph
        if line.startswith('-') or line.startswith('•') or line.startswith('★'):
            if current_paragraph:
                cleaned_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            cleaned_lines.append(line)
        # Check if previous line ended mid-sentence (lowercase continuation)
        elif current_paragraph and line[0].islower():
            current_paragraph.append(line)
        # Check if this looks like a continuation (doesn't start with capital after period)
        elif current_paragraph and not current_paragraph[-1].endswith(('.', '!', '?', ':')):
            current_paragraph.append(line)
        else:
            if current_paragraph:
                cleaned_lines.append(' '.join(current_paragraph))
            current_paragraph = [line]

    if current_paragraph:
        cleaned_lines.append(' '.join(current_paragraph))

    return '\n'.join(cleaned_lines)


def _build_cv_from_sections(sections: dict) -> str:
    """Build a clean CV text from parsed sections."""
    parts = []

    # Header
    if sections['name']:
        parts.append(f"# {sections['name']}")
    if sections['headline']:
        parts.append(sections['headline'])
    if sections['location']:
        parts.append(sections['location'])

    # Contact
    if sections['contact']:
        contact_parts = []
        if sections['contact'].get('email'):
            contact_parts.append(sections['contact']['email'])
        if sections['contact'].get('phone'):
            contact_parts.append(sections['contact']['phone'])
        if sections['contact'].get('linkedin'):
            contact_parts.append(sections['contact']['linkedin'])
        if sections['contact'].get('website'):
            contact_parts.append(sections['contact']['website'])
        if contact_parts:
            parts.append('\n' + ' | '.join(contact_parts))

    # Summary
    if sections['summary']:
        parts.append('\n## Summary\n' + sections['summary'])

    # Skills
    if sections['skills']:
        parts.append('\n## Skills\n' + ', '.join(sections['skills']))

    # Experience
    if sections['experience']:
        parts.append('\n## Experience')
        for exp in sections['experience']:
            exp_text = f"\n### {exp['title']}"
            if exp['company']:
                exp_text += f" at {exp['company']}"
            if exp['dates']:
                exp_text += f"\n{exp['dates']}"
            if exp['location']:
                exp_text += f" | {exp['location']}"
            if exp['description']:
                exp_text += f"\n{exp['description']}"
            parts.append(exp_text)

    # Education
    if sections['education']:
        parts.append('\n## Education')
        for edu in sections['education']:
            edu_text = f"\n### {edu['institution']}"
            if edu['degree']:
                edu_text += f"\n{edu['degree']}"
            if edu['dates']:
                edu_text += f" {edu['dates']}"
            if edu['details']:
                edu_text += f"\n{edu['details']}"
            parts.append(edu_text)

    # Languages
    if sections['languages']:
        parts.append('\n## Languages\n' + ', '.join(sections['languages']))

    # Certifications
    if sections['certifications']:
        parts.append('\n## Certifications\n' + '\n'.join(f"- {cert}" for cert in sections['certifications']))

    # Publications
    if sections['publications']:
        parts.append('\n## Publications\n' + '\n'.join(f"- {pub}" for pub in sections['publications']))

    return '\n'.join(parts)


def clean_extracted_text(text: str) -> str:
    """Clean up extracted PDF text (generic, non-LinkedIn).

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove page headers/footers that are just page numbers
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'Page \d+ of \d+\n?', '', text)

    # Clean up lines
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        # Skip very short lines that are likely artifacts
        if len(line) < 2 and not line.isalnum():
            continue
        lines.append(line)

    return '\n'.join(lines).strip()
