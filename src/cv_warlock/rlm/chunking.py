"""Smart chunking strategies for CVs and job specifications.

Parses documents by structure to enable targeted analysis.
"""

import re
from re import Pattern

from cv_warlock.rlm.models import CVChunks, JobChunks


class CVChunker:
    """Chunks CV text by document structure.

    Identifies sections like:
    - Contact Information
    - Summary/Objective
    - Work Experience (with individual jobs)
    - Education
    - Skills
    - Projects
    - Certifications
    """

    # Common CV section headers (case-insensitive patterns)
    SECTION_PATTERNS: dict[str, Pattern[str]] = {
        "contact": re.compile(
            r"^#+\s*(contact\s*(info|information)?|personal\s*(info|details)?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "summary": re.compile(
            r"^#+\s*(summary|professional\s+summary|profile|objective|about\s*me?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "experience": re.compile(
            r"^#+\s*(experience|work\s+experience|employment|professional\s+experience|work\s+history)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "education": re.compile(
            r"^#+\s*(education|academic|qualifications|degrees?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "skills": re.compile(
            r"^#+\s*(skills|technical\s+skills|core\s+competencies|competencies|expertise)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "projects": re.compile(
            r"^#+\s*(projects|personal\s+projects|key\s+projects|portfolio)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "certifications": re.compile(
            r"^#+\s*(certifications?|licenses?|credentials|professional\s+development)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "languages": re.compile(
            r"^#+\s*(languages?|spoken\s+languages?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "awards": re.compile(
            r"^#+\s*(awards?|honors?|achievements?|recognition)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "publications": re.compile(
            r"^#+\s*(publications?|papers?|research)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    }

    # Pattern for markdown headers of any level
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    # Pattern for job/position entries within experience
    JOB_ENTRY_PATTERN = re.compile(
        r"^###?\s*(.+?)\s*(?:at|@|,|\|)\s*(.+?)(?:\s*\||\s*\(|\s*$)",
        re.MULTILINE,
    )

    def chunk(self, cv_text: str) -> CVChunks:
        """Parse CV into structured chunks.

        Args:
            cv_text: Raw CV text (typically markdown).

        Returns:
            CVChunks with sections and individual items.
        """
        chunks = CVChunks(raw_text=cv_text)

        # Find all headers and their positions
        headers = list(self.HEADER_PATTERN.finditer(cv_text))

        if not headers:
            # No headers found, store entire text as summary
            chunks.summary = cv_text.strip()
            return chunks

        # Build sections dictionary from headers
        sections: dict[str, str] = {}
        for i, match in enumerate(headers):
            header_text = match.group(2).strip()
            start = match.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(cv_text)
            content = cv_text[start:end].strip()

            if content:
                sections[header_text.lower()] = content

        chunks.sections = sections

        # Extract known sections
        for section_key, pattern in self.SECTION_PATTERNS.items():
            for header_name, content in sections.items():
                if pattern.search(f"# {header_name}"):
                    self._assign_section(chunks, section_key, content)
                    break

        # Split experience into individual jobs
        if chunks.sections:
            for key, content in chunks.sections.items():
                if self.SECTION_PATTERNS["experience"].search(f"# {key}"):
                    chunks.experiences = self._split_experiences(content)
                    break

        # Split education into individual entries
        if chunks.sections:
            for key, content in chunks.sections.items():
                if self.SECTION_PATTERNS["education"].search(f"# {key}"):
                    chunks.education = self._split_by_subheaders(content)
                    break

        # Collect other sections not matched to known types
        known_keys = set()
        for pattern in self.SECTION_PATTERNS.values():
            for key in chunks.sections:
                if pattern.search(f"# {key}"):
                    known_keys.add(key)

        chunks.other_sections = {k: v for k, v in chunks.sections.items() if k not in known_keys}

        return chunks

    def _assign_section(self, chunks: CVChunks, section_key: str, content: str) -> None:
        """Assign content to the appropriate chunk field."""
        if section_key == "contact":
            chunks.contact = content
        elif section_key == "summary":
            chunks.summary = content
        elif section_key == "skills":
            chunks.skills = content
        elif section_key == "certifications":
            chunks.certifications = content
        # experience and education are handled separately as lists

    def _split_experiences(self, experience_text: str) -> list[str]:
        """Split experience section into individual job entries."""
        # Try to split by subheaders (### or ##)
        subheader_pattern = re.compile(r"^###?\s+", re.MULTILINE)
        parts = subheader_pattern.split(experience_text)

        if len(parts) > 1:
            # Found subheaders, clean up entries
            entries = []
            for part in parts:
                part = part.strip()
                if part and len(part) > 20:  # Skip very short fragments
                    entries.append(part)
            if entries:
                return entries

        # No subheaders, try splitting by blank lines + date patterns
        date_pattern = re.compile(
            r"\n\n+(?=.*?\d{4})",  # Blank lines followed by a year
            re.MULTILINE,
        )
        parts = date_pattern.split(experience_text)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]

        # Fallback: return entire section as one chunk
        return [experience_text.strip()] if experience_text.strip() else []

    def _split_by_subheaders(self, section_text: str) -> list[str]:
        """Split a section by subheaders into individual entries."""
        subheader_pattern = re.compile(r"^###?\s+", re.MULTILINE)
        parts = subheader_pattern.split(section_text)

        entries = []
        for part in parts:
            part = part.strip()
            if part:
                entries.append(part)

        return entries if entries else [section_text.strip()]

    def get_section(self, chunks: CVChunks, section: str) -> str | None:
        """Retrieve a specific section by name.

        Args:
            chunks: Parsed CV chunks.
            section: Section name (case-insensitive).

        Returns:
            Section content or None.
        """
        section_lower = section.lower()

        # Check direct attributes first
        if section_lower == "summary":
            return chunks.summary
        elif section_lower == "skills":
            return chunks.skills
        elif section_lower == "contact":
            return chunks.contact
        elif section_lower == "certifications":
            return chunks.certifications

        # Check sections dict
        return chunks.sections.get(section_lower)


class JobChunker:
    """Chunks job spec by document structure.

    Identifies sections like:
    - Job Title/Overview
    - Requirements (required vs preferred)
    - Responsibilities
    - Benefits
    - Company Information
    """

    # Common job spec section headers
    SECTION_PATTERNS: dict[str, Pattern[str]] = {
        "overview": re.compile(
            r"^#+\s*(overview|about\s+the\s+role|position\s+summary|job\s+description|the\s+role)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "requirements": re.compile(
            r"^#+\s*(requirements?|qualifications?|what\s+you.+need|must\s+have|required)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "preferred": re.compile(
            r"^#+\s*(preferred|nice\s+to\s+have|bonus|desired|ideal)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "responsibilities": re.compile(
            r"^#+\s*(responsibilities|duties|what\s+you.+do|your\s+role|key\s+accountabilities)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "benefits": re.compile(
            r"^#+\s*(benefits?|perks?|what\s+we\s+offer|compensation|package)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "company": re.compile(
            r"^#+\s*(about\s+(us|the\s+company)|company|who\s+we\s+are|our\s+company)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    }

    # Pattern for extracting bullet points
    BULLET_PATTERN = re.compile(r"^[\*\-\+•]\s*(.+)$", re.MULTILINE)

    # Pattern for markdown headers
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def chunk(self, job_text: str) -> JobChunks:
        """Parse job spec into structured chunks.

        Args:
            job_text: Raw job specification text.

        Returns:
            JobChunks with sections and requirements.
        """
        chunks = JobChunks(raw_text=job_text)

        # Try to extract job title from first line or first header
        first_line = job_text.strip().split("\n")[0]
        if first_line.startswith("#"):
            chunks.title = first_line.lstrip("#").strip()
        else:
            chunks.title = first_line[:100]  # Truncate if too long

        # Find all headers and their positions
        headers = list(self.HEADER_PATTERN.finditer(job_text))

        if not headers:
            # No headers, store entire text as overview
            chunks.overview = job_text.strip()
            return chunks

        # Build sections dictionary
        sections: dict[str, str] = {}
        for i, match in enumerate(headers):
            header_text = match.group(2).strip()
            start = match.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(job_text)
            content = job_text[start:end].strip()

            if content:
                sections[header_text.lower()] = content

        chunks.sections = sections

        # Extract known sections
        for header_name, content in sections.items():
            header_with_prefix = f"# {header_name}"

            if self.SECTION_PATTERNS["overview"].search(header_with_prefix):
                chunks.overview = content
            elif self.SECTION_PATTERNS["requirements"].search(header_with_prefix):
                chunks.required_qualifications = self._extract_bullets(content)
            elif self.SECTION_PATTERNS["preferred"].search(header_with_prefix):
                chunks.preferred_qualifications = self._extract_bullets(content)
            elif self.SECTION_PATTERNS["responsibilities"].search(header_with_prefix):
                chunks.responsibilities = self._extract_bullets(content)
            elif self.SECTION_PATTERNS["benefits"].search(header_with_prefix):
                chunks.benefits = content
            elif self.SECTION_PATTERNS["company"].search(header_with_prefix):
                chunks.company_info = content

        # If no explicit required/preferred split, try to infer from content
        if not chunks.required_qualifications:
            for key, content in sections.items():
                if any(word in key.lower() for word in ["qualif", "require", "need", "must"]):
                    bullets = self._extract_bullets(content)
                    chunks.required_qualifications = self._classify_requirements(
                        bullets, required=True
                    )
                    chunks.preferred_qualifications = self._classify_requirements(
                        bullets, required=False
                    )
                    break

        # Collect other sections
        known_keys = set()
        for pattern in self.SECTION_PATTERNS.values():
            for key in sections:
                if pattern.search(f"# {key}"):
                    known_keys.add(key)

        chunks.other_sections = {k: v for k, v in sections.items() if k not in known_keys}

        return chunks

    def _extract_bullets(self, text: str) -> list[str]:
        """Extract bullet points from text.

        Args:
            text: Section text containing bullet points.

        Returns:
            List of bullet point contents.
        """
        matches = self.BULLET_PATTERN.findall(text)
        if matches:
            return [m.strip() for m in matches if m.strip()]

        # No bullet pattern, try splitting by newlines
        lines = text.strip().split("\n")
        return [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]

    def _classify_requirements(self, bullets: list[str], required: bool) -> list[str]:
        """Classify requirements as required or preferred.

        Args:
            bullets: List of requirement strings.
            required: If True, return required items; else return preferred.

        Returns:
            Filtered list of requirements.
        """
        required_keywords = ["must", "required", "essential", "need", "minimum"]
        preferred_keywords = ["nice", "preferred", "bonus", "ideal", "plus", "advantage"]

        result = []
        for bullet in bullets:
            bullet_lower = bullet.lower()

            is_required = any(kw in bullet_lower for kw in required_keywords)
            is_preferred = any(kw in bullet_lower for kw in preferred_keywords)

            if required:
                # Include if explicitly required OR not explicitly preferred
                if is_required or not is_preferred:
                    result.append(bullet)
            else:
                # Include if explicitly preferred
                if is_preferred:
                    result.append(bullet)

        return result

    def get_requirements_by_priority(
        self, chunks: JobChunks, top_n: int | None = None
    ) -> list[tuple[str, bool]]:
        """Get requirements sorted by priority.

        Args:
            chunks: Parsed job chunks.
            top_n: If provided, return only top N requirements.

        Returns:
            List of (requirement, is_required) tuples.
        """
        requirements = [(req, True) for req in chunks.required_qualifications] + [
            (req, False) for req in chunks.preferred_qualifications
        ]

        if top_n:
            return requirements[:top_n]
        return requirements
