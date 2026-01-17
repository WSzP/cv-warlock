"""LinkedIn profile fetcher utility.

Supports multiple extraction strategies:
1. Jina Reader API (free, no key needed) - Best for JavaScript-rendered pages
2. Direct HTTP fetch with BeautifulSoup - Fallback for simple cases
"""

import json
import os
import re

import httpx


def extract_linkedin_username(url: str) -> str | None:
    """Extract LinkedIn username from URL.

    Args:
        url: LinkedIn profile URL.

    Returns:
        Username if valid LinkedIn URL, None otherwise.
    """
    pattern = r"(?:https?://)?(?:www\.)?(?:\w+\.)?linkedin\.com/in/([^/?\s]+)"
    match = re.search(pattern, url.strip())
    return match.group(1) if match else None


def fetch_linkedin_profile(url: str) -> tuple[str | None, str | None]:
    """Fetch LinkedIn profile and extract CV content.

    Tries multiple strategies in order:
    1. Jina Reader API (free, handles JavaScript)
    2. Direct HTTP fetch (fallback)

    Args:
        url: LinkedIn profile URL.

    Returns:
        Tuple of (cv_text, error_message). One will be None.
    """
    username = extract_linkedin_username(url)
    if not username:
        return None, "Invalid LinkedIn URL. Please use format: linkedin.com/in/username"

    profile_url = f"https://www.linkedin.com/in/{username}/"

    # Strategy 1: Try Jina Reader API (free, handles JavaScript rendering)
    cv_text, error = fetch_via_jina_reader(profile_url, username)
    if cv_text:
        return cv_text, None

    # Strategy 2: Try direct HTTP fetch
    cv_text, error = fetch_via_direct_http(profile_url, username)
    if cv_text:
        return cv_text, None

    # All strategies failed
    return None, (
        f"Could not extract full profile data. {error or ''}\n\n"
        "**Recommended alternatives:**\n"
        "1. Use LinkedIn's 'Save to PDF' feature (Profile → More → Save to PDF)\n"
        "2. Copy your profile text manually\n"
        "3. Switch to 'Paste Text' mode and enter your CV"
    )


def fetch_via_jina_reader(url: str, username: str) -> tuple[str | None, str | None]:
    """Fetch LinkedIn profile using Jina Reader API.

    Jina Reader (r.jina.ai) is a free API that converts any URL to
    LLM-friendly markdown, handling JavaScript rendering.

    Args:
        url: LinkedIn profile URL.
        username: LinkedIn username.

    Returns:
        Tuple of (cv_text, error_message).
    """
    jina_url = f"https://r.jina.ai/{url}"

    try:
        headers = {
            "Accept": "text/plain",
            "User-Agent": "CV-Warlock/1.0",
        }

        # Add Jina API key if available (for higher rate limits)
        jina_key = os.getenv("JINA_API_KEY")
        if jina_key:
            headers["Authorization"] = f"Bearer {jina_key}"

        with httpx.Client(timeout=60.0) as client:
            response = client.get(jina_url, headers=headers)
            response.raise_for_status()

            content = response.text

            # Check if we got meaningful content
            if len(content) < 200:
                return None, "Jina Reader returned insufficient content"

            # Check for login wall indicators
            if "sign in" in content.lower()[:500] and "authwall" in content.lower():
                return None, "LinkedIn requires authentication"

            # Format the content as a CV
            cv_text = format_jina_content_as_cv(content, username)

            if cv_text and len(cv_text) > 300:
                return cv_text, None

            return None, "Could not extract meaningful profile data from Jina Reader"

    except httpx.TimeoutException:
        return None, "Jina Reader request timed out"
    except httpx.HTTPStatusError as e:
        return None, f"Jina Reader error: HTTP {e.response.status_code}"
    except Exception as e:
        return None, f"Jina Reader error: {str(e)}"


def format_jina_content_as_cv(content: str, username: str) -> str | None:
    """Format Jina Reader content as a structured CV.

    Args:
        content: Raw markdown content from Jina Reader.
        username: LinkedIn username.

    Returns:
        Formatted CV text or None.
    """
    # Jina Reader typically returns well-structured markdown
    # We just need to clean it up and ensure it has proper sections

    lines = content.split("\n")
    cleaned_lines = []
    in_content = False

    for line in lines:
        # Skip Jina metadata and navigation elements
        if line.startswith("URL Source:") or line.startswith("Markdown Content:"):
            in_content = True
            continue

        if not in_content and not line.strip():
            continue

        # Skip common LinkedIn navigation/UI elements
        skip_patterns = [
            "Skip to main content",
            "Sign in",
            "Join now",
            "LinkedIn",
            "Get the app",
            "More from LinkedIn",
            "Cookie Policy",
            "Privacy Policy",
            "User Agreement",
            "Agree & Join",
        ]

        if any(pattern.lower() in line.lower() for pattern in skip_patterns):
            continue

        # Skip image alt text and links that are just URLs
        if line.startswith("![") or (line.startswith("[") and "](http" in line):
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()

    # Add LinkedIn reference if not present
    if "linkedin.com" not in result.lower():
        result = f"{result}\n\nLinkedIn: linkedin.com/in/{username}"

    return result if len(result) > 200 else None


def fetch_via_direct_http(url: str, username: str) -> tuple[str | None, str | None]:
    """Fetch LinkedIn profile via direct HTTP request.

    This is a fallback that works for some public profiles but often
    returns limited data due to JavaScript rendering requirements.

    Args:
        url: LinkedIn profile URL.
        username: LinkedIn username.

    Returns:
        Tuple of (cv_text, error_message).
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

            html = response.text

            # Check for login wall
            if "authwall" in response.url.path:
                return None, "LinkedIn requires authentication"

            cv_text, is_partial = parse_linkedin_html(html, username)

            if cv_text:
                if is_partial:
                    cv_text += (
                        "\n\n---\n"
                        "*Note: Only partial data was retrieved. "
                        "Please complete the missing sections above.*"
                    )
                return cv_text, None

            return None, "Could not parse profile HTML"

    except httpx.TimeoutException:
        return None, "Request timed out"
    except httpx.HTTPStatusError as e:
        return None, f"HTTP error: {e.response.status_code}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def parse_linkedin_html(html: str, username: str) -> tuple[str | None, bool]:
    """Parse LinkedIn HTML and extract profile information.

    Args:
        html: Raw HTML content.
        username: LinkedIn username.

    Returns:
        Tuple of (CV text, is_partial).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None, True

    soup = BeautifulSoup(html, "html.parser")
    sections = []
    has_experience = False
    has_education = False

    # Extract from JSON-LD first
    json_ld = extract_json_ld(soup)

    # Name
    name = None
    if json_ld and json_ld.get("name"):
        name = json_ld["name"]
    else:
        og_title = soup.select_one('meta[property="og:title"]')
        if og_title:
            name = og_title.get("content", "").split(" - ")[0].split(" | ")[0].strip()

    sections.append(f"# {name or f'LinkedIn Profile: {username}'}")

    # Headline
    headline = None
    if json_ld and json_ld.get("jobTitle"):
        headline = json_ld["jobTitle"]
    else:
        meta_desc = soup.select_one('meta[name="description"]')
        if meta_desc:
            headline = meta_desc.get("content", "").strip()

    if headline and len(headline) < 300:
        sections.append(f"**{headline}**")

    sections.append("")
    sections.append(f"LinkedIn: linkedin.com/in/{username}")
    sections.append("")

    # Summary
    summary = json_ld.get("description") if json_ld else None
    if not summary:
        summary = extract_about_from_scripts(html)

    if summary and len(summary) > 20:
        sections.append("## Summary")
        sections.append(re.sub(r'\s+', ' ', summary).strip()[:2000])
        sections.append("")

    # Experience from JSON-LD
    if json_ld and json_ld.get("worksFor"):
        works_for = json_ld["worksFor"]
        if isinstance(works_for, list) and works_for:
            sections.append("## Experience")
            has_experience = True
            for job in works_for[:10]:
                if isinstance(job, dict) and job.get("name"):
                    sections.append(f"**{job['name']}**")
                    sections.append("")

    # Education from JSON-LD
    if json_ld and json_ld.get("alumniOf"):
        alumni = json_ld["alumniOf"]
        if isinstance(alumni, list) and alumni:
            sections.append("## Education")
            has_education = True
            for school in alumni[:5]:
                if isinstance(school, dict) and school.get("name"):
                    sections.append(f"**{school['name']}**")
                    sections.append("")

    # Add placeholders for missing sections
    if not has_experience:
        sections.append("## Experience")
        sections.append("*[Add your work experience here]*")
        sections.append("")

    if not has_education:
        sections.append("## Education")
        sections.append("*[Add your education here]*")
        sections.append("")

    sections.append("## Skills")
    sections.append("*[Add your skills here]*")
    sections.append("")

    result = "\n".join(sections)
    is_partial = not has_experience or not has_education

    return (result, is_partial) if len(result) > 100 else (None, True)


def extract_json_ld(soup) -> dict | None:
    """Extract JSON-LD structured data."""
    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        try:
            data = json.loads(script.string or "")
            if isinstance(data, dict) and data.get("@type") == "Person":
                return data
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("@type") == "Person":
                        return item
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def extract_about_from_scripts(html: str) -> str | None:
    """Extract About section from inline scripts."""
    patterns = [
        r'"summary"\s*:\s*"([^"]{20,2000})"',
        r'"about"\s*:\s*"([^"]{20,2000})"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            text = match.group(1)
            try:
                text = json.loads(f'"{text}"')
            except json.JSONDecodeError:
                pass
            return text
    return None
