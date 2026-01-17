"""LinkedIn profile fetcher utility."""

import re

import httpx


def extract_linkedin_username(url: str) -> str | None:
    """Extract LinkedIn username from URL.

    Args:
        url: LinkedIn profile URL.

    Returns:
        Username if valid LinkedIn URL, None otherwise.
    """
    # Match patterns like linkedin.com/in/username or www.linkedin.com/in/username
    pattern = r"(?:https?://)?(?:www\.)?(?:\w+\.)?linkedin\.com/in/([^/?\s]+)"
    match = re.search(pattern, url.strip())
    return match.group(1) if match else None


def fetch_linkedin_profile(url: str) -> tuple[str | None, str | None]:
    """Fetch LinkedIn profile and extract CV content.

    Args:
        url: LinkedIn profile URL.

    Returns:
        Tuple of (cv_text, error_message). One will be None.
    """
    username = extract_linkedin_username(url)
    if not username:
        return None, "Invalid LinkedIn URL. Please use format: linkedin.com/in/username"

    # Normalize URL
    profile_url = f"https://www.linkedin.com/in/{username}/"

    try:
        # Use browser-like headers to avoid blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            response = client.get(profile_url, headers=headers)
            response.raise_for_status()

            html_content = response.text

            # Check if we got a login wall
            if "authwall" in response.url.path or "login" in html_content.lower()[:1000]:
                return None, (
                    "LinkedIn requires authentication for this profile. "
                    "Please use LinkedIn's 'Save to PDF' feature to export your CV, "
                    "then paste the content here."
                )

            # Parse the HTML to extract profile data
            cv_text = parse_linkedin_html(html_content, username)

            if cv_text:
                return cv_text, None
            else:
                return None, (
                    "Could not extract profile data. The profile may be private or "
                    "LinkedIn's format may have changed. Please paste your CV manually."
                )

    except httpx.TimeoutException:
        return None, "Request timed out. Please try again or paste your CV manually."
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None, f"Profile not found: {username}"
        elif e.response.status_code == 429:
            return None, "Too many requests. Please wait a moment and try again."
        else:
            return None, f"Failed to fetch profile (HTTP {e.response.status_code})"
    except Exception as e:
        return None, f"Error fetching profile: {str(e)}"


def parse_linkedin_html(html: str, username: str) -> str | None:
    """Parse LinkedIn HTML and extract profile information.

    This is a basic parser for public LinkedIn profiles.
    LinkedIn's HTML structure changes frequently, so this may need updates.

    Args:
        html: Raw HTML content.
        username: LinkedIn username for fallback.

    Returns:
        Formatted CV text or None if parsing fails.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None

    soup = BeautifulSoup(html, "html.parser")

    sections = []

    # Try to extract name from various possible locations
    name = None
    name_selectors = [
        "h1.top-card-layout__title",
        "h1.text-heading-xlarge",
        'meta[property="og:title"]',
    ]
    for selector in name_selectors:
        if selector.startswith("meta"):
            elem = soup.select_one(selector)
            if elem:
                name = elem.get("content", "").split("|")[0].strip()
                break
        else:
            elem = soup.select_one(selector)
            if elem:
                name = elem.get_text(strip=True)
                break

    if name:
        sections.append(f"# {name}")
    else:
        sections.append(f"# LinkedIn Profile: {username}")

    # Try to extract headline/title
    headline_selectors = [
        "h2.top-card-layout__headline",
        "div.text-body-medium",
        'meta[name="description"]',
    ]
    for selector in headline_selectors:
        if selector.startswith("meta"):
            elem = soup.select_one(selector)
            if elem:
                headline = elem.get("content", "").strip()
                if headline and len(headline) < 200:
                    sections.append(f"**{headline}**")
                    break
        else:
            elem = soup.select_one(selector)
            if elem:
                headline = elem.get_text(strip=True)
                if headline and len(headline) < 200:
                    sections.append(f"**{headline}**")
                    break

    sections.append("")
    sections.append(f"LinkedIn: linkedin.com/in/{username}")
    sections.append("")

    # Extract about/summary section
    about_section = soup.select_one("section.summary, div.pv-about-section")
    if about_section:
        about_text = about_section.get_text(strip=True)
        if about_text:
            sections.append("## Summary")
            sections.append(about_text[:1000])  # Limit length
            sections.append("")

    # Look for experience section
    exp_section = soup.select_one("section.experience, div.pv-experience-section")
    if exp_section:
        sections.append("## Experience")
        positions = exp_section.select("li.pv-entity__position-group-pager, div.pv-position-entity")
        for pos in positions[:10]:  # Limit to 10 positions
            title = pos.select_one("h3, .pv-entity__secondary-title")
            company = pos.select_one("p.pv-entity__secondary-title, .pv-entity__company-summary-info")
            if title:
                sections.append(f"### {title.get_text(strip=True)}")
            if company:
                sections.append(f"**{company.get_text(strip=True)}**")
            sections.append("")

    # Look for education section
    edu_section = soup.select_one("section.education, div.pv-education-section")
    if edu_section:
        sections.append("## Education")
        schools = edu_section.select("li.pv-education-entity, div.pv-entity__degree-info")
        for school in schools[:5]:
            school_name = school.select_one("h3, .pv-entity__school-name")
            degree = school.select_one("p, .pv-entity__degree-name")
            if school_name:
                sections.append(f"**{school_name.get_text(strip=True)}**")
            if degree:
                sections.append(degree.get_text(strip=True))
            sections.append("")

    # Look for skills
    skills_section = soup.select_one("section.skills, div.pv-skill-categories-section")
    if skills_section:
        sections.append("## Skills")
        skills = skills_section.select("span.pv-skill-category-entity__name-text, li.pv-skill-entity")
        skill_names = [s.get_text(strip=True) for s in skills[:20]]
        if skill_names:
            sections.append(", ".join(skill_names))
            sections.append("")

    result = "\n".join(sections)

    # Only return if we got meaningful content
    if len(result) > 100:
        return result

    return None
