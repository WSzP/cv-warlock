"""Streamlit web UI for CV Warlock."""

import os
import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env.local
from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env.local")

import streamlit as st  # noqa: E402
from components.cv_input import render_cv_input  # noqa: E402
from components.job_input import render_job_input  # noqa: E402
from components.result_display import render_result  # noqa: E402


def get_env_api_key(provider: str) -> str | None:
    """Get API key from environment for the given provider."""
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    elif provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    elif provider == "google":
        return os.getenv("GOOGLE_API_KEY")
    return None


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


def parse_error_details(error_message: str, provider: str, model: str) -> dict:
    """Parse error message and return detailed diagnostic information.

    Returns dict with:
        - category: Error category (connection, auth, rate_limit, model, parsing, unknown)
        - title: Short error title
        - explanation: What went wrong
        - cause: Likely cause
        - solution: How to fix it
        - technical: Technical details for debugging
    """
    error_lower = error_message.lower()

    # RLM timeout errors (check first as they contain "timeout")
    if "rlm timeout" in error_lower:
        return {
            "category": "rlm_timeout",
            "title": "RLM Analysis Timeout",
            "explanation": "The recursive analysis took too long and was stopped.",
            "cause": "Large or complex documents require more processing time than the limit allows.",
            "solution": [
                "The system automatically fell back to standard extraction",
                "Your CV should still be processed successfully",
                "For very large documents, consider splitting them or simplifying formatting",
                "Try again - API response times vary",
            ],
            "technical": error_message,
        }

    # Connection errors
    if any(x in error_lower for x in ["connection", "connect", "network", "timeout", "timed out"]):
        return {
            "category": "connection",
            "title": "Connection Failed",
            "explanation": f"Could not connect to the {provider.title()} API servers.",
            "cause": "Network issues, firewall blocking, or the API service is down.",
            "solution": [
                "Check your internet connection",
                "Try again in a few moments",
                f"Check {provider.title()} status page for outages",
                "If using VPN/proxy, try disabling it",
            ],
            "technical": error_message,
        }

    # Authentication errors
    if any(
        x in error_lower
        for x in [
            "unauthorized",
            "401",
            "invalid api key",
            "authentication",
            "invalid_api_key",
            "invalid x-api-key",
        ]
    ):
        return {
            "category": "auth",
            "title": "Authentication Failed",
            "explanation": f"Your {provider.title()} API key was rejected.",
            "cause": "The API key is invalid, expired, or doesn't have proper permissions.",
            "solution": [
                f"Verify your {provider.title()} API key is correct",
                "Check if the key has expired",
                "Ensure the key has the required permissions",
                "Generate a new API key if needed",
            ],
            "technical": error_message,
        }

    # Rate limit errors
    if any(
        x in error_lower
        for x in ["rate limit", "rate_limit", "429", "too many requests", "quota", "exceeded"]
    ):
        return {
            "category": "rate_limit",
            "title": "Rate Limit Exceeded",
            "explanation": f"Too many requests to {provider.title()} API.",
            "cause": "You've exceeded the API rate limit or your usage quota.",
            "solution": [
                "Wait a few minutes before trying again",
                "Check your API usage dashboard",
                "Consider upgrading your API plan",
                "Try using a different model",
            ],
            "technical": error_message,
        }

    # Model errors
    if any(
        x in error_lower
        for x in [
            "model not found",
            "invalid model",
            "model_not_found",
            "does not exist",
            "not available",
        ]
    ):
        return {
            "category": "model",
            "title": "Model Not Available",
            "explanation": f"The model '{model}' is not available.",
            "cause": "The model name is incorrect or not accessible with your API key.",
            "solution": [
                "Select a different model from the dropdown",
                "Check if the model name is spelled correctly",
                "Verify your API plan includes access to this model",
            ],
            "technical": error_message,
        }

    # Context/token limit errors
    if any(
        x in error_lower for x in ["context length", "token limit", "too long", "maximum context"]
    ):
        return {
            "category": "context",
            "title": "Input Too Long",
            "explanation": "Your CV or job spec is too long for the model to process.",
            "cause": "The combined length exceeds the model's context window.",
            "solution": [
                "Try shortening your CV (remove less relevant sections)",
                "Use a shorter job specification",
                "Try a model with a larger context window",
            ],
            "technical": error_message,
        }

    # Parsing/extraction errors
    if any(
        x in error_lower
        for x in [
            "extraction failed",
            "parsing",
            "parse error",
            "invalid json",
            "failed to extract",
        ]
    ):
        step = "unknown"
        if "cv extraction" in error_lower:
            step = "CV"
        elif "job extraction" in error_lower:
            step = "job specification"

        return {
            "category": "parsing",
            "title": f"{step.title()} Parsing Failed",
            "explanation": f"Failed to parse and understand your {step}.",
            "cause": "The document format may be unusual or contain unsupported elements.",
            "solution": [
                f"Check that your {step} is in a standard format",
                "Remove any unusual formatting or special characters",
                "Try pasting plain text instead of formatted text",
                "Ensure the content is in English",
            ],
            "technical": error_message,
        }

    # Server errors
    if any(
        x in error_lower for x in ["500", "502", "503", "504", "server error", "internal error"]
    ):
        return {
            "category": "server",
            "title": "API Server Error",
            "explanation": f"The {provider.title()} API server encountered an error.",
            "cause": "Temporary server-side issue.",
            "solution": [
                "Wait a moment and try again",
                f"Check {provider.title()} status page",
                "Try a different model",
            ],
            "technical": error_message,
        }

    # Default/unknown errors
    return {
        "category": "unknown",
        "title": "Processing Error",
        "explanation": "An unexpected error occurred during CV tailoring.",
        "cause": "Unknown - see technical details below.",
        "solution": [
            "Try again",
            "Check your inputs are valid",
            "Try a different model or provider",
            "Report this issue if it persists",
        ],
        "technical": error_message,
    }


def render_error_details(error_info: dict, elapsed_time: float | None = None):
    """Render detailed error information in Streamlit."""
    st.error(f"**{error_info['title']}**")

    # Show elapsed time if available
    if elapsed_time is not None:
        st.markdown(f"*Failed after {format_elapsed_time(elapsed_time)}*")

    # Main explanation
    st.markdown(f"**What happened:** {error_info['explanation']}")
    st.markdown(f"**Likely cause:** {error_info['cause']}")

    # Solutions
    st.markdown("**How to fix it:**")
    for solution in error_info["solution"]:
        st.markdown(f"- {solution}")

    # Technical details in expander
    with st.expander("Technical Details", expanded=False):
        st.code(error_info["technical"], language=None)


# Ordered workflow steps for checklist display
WORKFLOW_STEPS = [
    ("graph_init", "Initializing graph"),
    ("validate_inputs", "Initializing workflow"),
    ("extract_cv", "Extracting CV structure"),
    ("extract_job", "Analyzing job requirements"),
    ("analyze_match", "Matching profile to requirements"),
    ("create_plan", "Creating tailoring strategy"),
    ("tailor_skills", "Tailoring skills section"),
    ("tailor_experiences", "Tailoring work experiences"),
    ("tailor_summary", "Crafting professional summary"),
    ("assemble_cv", "Assembling final CV"),
]


def render_step_checklist(
    completed_steps: set[str],
    current_step: str | None,
    failed_step: str | None,
    current_description: str | None = None,
) -> str:
    """Render the step checklist as HTML with status icons.

    Args:
        completed_steps: Set of step names that have completed successfully
        current_step: The step currently being processed (or None)
        failed_step: The step that failed (or None)
        current_description: Optional description for the current step

    Returns:
        HTML string for the checklist
    """
    lines = []
    for step_name, step_label in WORKFLOW_STEPS:
        if step_name in completed_steps:
            # Completed - green checkmark
            icon = '<span style="color: #28a745;">✓</span>'
            style = "color: #28a745;"
        elif step_name == failed_step:
            # Failed - red X
            icon = '<span style="color: #dc3545;">✗</span>'
            style = "color: #dc3545; font-weight: bold;"
        elif step_name == current_step:
            # In progress - spinning indicator
            icon = '<span class="step-spinner">◐</span>'
            style = "color: #007bff; font-weight: bold;"
            # Use custom description if provided
            if current_description:
                step_label = current_description.rstrip(".")
        else:
            # Pending - empty circle
            icon = '<span style="color: #6c757d;">○</span>'
            style = "color: #6c757d;"

        lines.append(f'<div style="margin: 4px 0; {style}">{icon} {step_label}</div>')

    return "\n".join(lines)


# Page configuration - use favicon from favicon-pack
favicon_path = project_root / "favicon-pack" / "favicon.ico"
st.set_page_config(
    page_title="CV Warlock",
    page_icon=str(favicon_path) if favicon_path.exists() else ":magic_wand:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar logo
logo_for_sidebar = project_root / "assets/cv-warlock-logo-small.webp"
if logo_for_sidebar.exists():
    st.logo(image=str(logo_for_sidebar))

# Custom CSS
st.markdown(
    """
    <style>
    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 1rem 0 1.5rem 0;
    }
    .header-logo {
        max-width: 280px;
        width: 100%;
        height: auto;
        margin-bottom: 0.5rem;
    }
    .header-tagline {
        font-size: 1.15rem;
        color: #666;
        margin: 0;
        font-weight: 400;
    }
    /* Dark mode support for tagline */
    @media (prefers-color-scheme: dark) {
        .header-tagline {
            color: #a0a0a0;
        }
    }
    .stTextArea textarea {
        font-family: monospace;
    }
    .timing-display {
        font-family: monospace;
        font-size: 1.1rem;
        color: #0066cc;
        padding: 0.5rem;
        background: #f0f8ff;
        border-radius: 4px;
        margin-top: 0.5rem;
    }
    .realtime-timer {
        font-family: monospace;
        font-size: 1.1rem;
        font-weight: bold;
        color: #0066cc;
    }
    .api-time {
        font-family: monospace;
        font-size: 0.95rem;
        color: #666;
    }
    /* Step checklist styles */
    .step-checklist {
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    @keyframes spin {
        0% { content: "◐"; }
        25% { content: "◓"; }
        50% { content: "◑"; }
        75% { content: "◒"; }
    }
    .step-spinner {
        display: inline-block;
        animation: rotate 1s linear infinite;
        color: #007bff;
    }
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Main Streamlit application."""
    # Initialize session state for two-phase processing
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "process_start_time" not in st.session_state:
        st.session_state.process_start_time = None

    # Sidebar configuration
    with st.sidebar:
        st.title("AI Provider")

        provider = st.selectbox(
            "LLM Provider",
            options=["anthropic", "openai", "google"],
            index=0,
            help="Choose the AI provider. Model is auto-selected via Dual-Model Strategy.",
        )

        st.caption(
            "**Dual-Model Strategy:** Model is auto-selected based on provider. "
            "RLM uses a stronger model for orchestration and a faster one for sub-calls."
        )

        # Check if API key exists in environment
        env_api_key = get_env_api_key(provider)

        # Only show API key input if not in environment
        if env_api_key:
            st.success(f"{provider.title()} API key loaded from .env.local")
            api_key = None  # Will use env key

            # Test API key button
            if st.button("Test API Key", key="test_api_key"):
                with st.spinner(f"Testing {provider.title()} API key..."):
                    try:
                        # Import test functions
                        sys.path.insert(0, str(project_root / "scripts"))
                        from test_api_keys import (
                            test_anthropic_key,
                            test_google_key,
                            test_openai_key,
                        )

                        if provider == "openai":
                            success, message = test_openai_key(env_api_key)
                        elif provider == "anthropic":
                            success, message = test_anthropic_key(env_api_key)
                        elif provider == "google":
                            success, message = test_google_key(env_api_key)
                        else:
                            success, message = False, f"Unknown provider: {provider}"

                        if success:
                            st.success(f"✓ {message}")
                        else:
                            st.error(f"✗ {message}")
                    except Exception as e:
                        st.error(f"Test failed: {e}")
        else:
            api_key = st.text_input(
                f"{provider.title()} API Key",
                type="password",
                help="Enter your API key (or set it in .env.local)",
            )

        st.divider()

        # Experience lookback settings
        st.subheader("Experience Lookback")

        lookback_years = st.slider(
            "Years to tailor",
            min_value=1,
            max_value=20,
            value=4,
            step=1,
            help="Only tailor job experiences that ended within this many years. "
            "Older jobs will be included but not modified.",
            key="lookback_years",
        )

        from datetime import datetime

        current_year = datetime.now().year
        cutoff_year = current_year - lookback_years

        if lookback_years <= 3:
            st.warning(
                f"**Short lookback ({lookback_years} years):** Only jobs ending after "
                f"{cutoff_year} will be tailored. Older jobs pass through unchanged. "
                "Use for roles where recent experience matters most."
            )
        elif lookback_years <= 7:
            st.info(
                f"**Standard lookback ({lookback_years} years):** Jobs ending after "
                f"{cutoff_year} will be tailored to emphasize relevant skills. "
                "Older jobs are included but not modified."
            )
        else:
            st.success(
                f"**Extended lookback ({lookback_years} years):** Most of your career "
                f"history (jobs ending after {cutoff_year}) will be tailored. "
                "Good for senior roles valuing broad experience."
            )

        st.divider()

        # RLM (Recursive Language Model) settings
        st.subheader("RLM Mode")

        st.checkbox(
            "Enable RLM",
            value=True,
            help="Use Recursive Language Model for handling large CVs and job specs",
            key="use_rlm",
        )

        st.caption(
            "**Dual-Model Strategy:** RLM uses a stronger model for "
            "orchestration and a faster model for sub-calls. "
            "This enables processing of arbitrarily long documents with interpretable reasoning."
        )

        st.divider()

        # Cover Letter settings
        st.subheader("Cover Letter")

        st.slider(
            "Character limit",
            min_value=500,
            max_value=5000,
            value=2000,
            step=250,
            help="Target length for generated cover letter",
            key="cover_letter_char_limit",
        )

        # Show guidance based on character limit
        char_limit = st.session_state.get("cover_letter_char_limit", 2000)
        if char_limit < 1500:
            st.caption(
                "**Short format:** Best for quick applications or character-limited forms. "
                "Will focus on 2-3 key points."
            )
        elif char_limit < 3000:
            st.caption(
                "**Standard format:** Ideal for most applications. "
                "Balances detail with readability."
            )
        else:
            st.caption(
                "**Extended format:** For applications allowing longer responses. "
                "Includes more context and achievements."
            )

        st.divider()

        # LangSmith tracing status
        langsmith_enabled = (
            os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_TRACING", "").lower() == "true"
        )
        if langsmith_enabled:
            project = os.getenv("LANGSMITH_PROJECT", "cv-warlock")
            endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
            # Determine the dashboard URL based on endpoint
            if "eu.api.smith" in endpoint:
                dashboard_url = "https://eu.smith.langchain.com/"
            else:
                dashboard_url = "https://smith.langchain.com/"
            st.success(f"LangSmith tracing: **{project}**")
            st.markdown(f"[View traces]({dashboard_url})")

        st.caption(
            "🧙 This langchain experiment was created by [Peter W. Szabo](https://www.linkedin.com/in/wszabopeter/)."
        )

    # Main content - Header with logo and title
    logo_path = project_root / "assets/cv-warlock-logo-small.webp"

    # Logo on left, title + tagline on right
    if logo_path.exists():
        logo_col, title_col = st.columns([1, 11], gap="small")
        with logo_col:
            st.image(str(logo_path))
        with title_col:
            st.markdown(
                '<h1 style="margin: 0; padding: 0;">'
                '<span style="color: #a9e53f;">CV</span> '
                '<span style="color: #4b2d73;">Warlock</span></h1>'
                '<p class="header-tagline" style="margin: 0;">AI-powered CV tailoring for job applications</p>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<h1><span style="color: #a9e53f;">CV</span> '
            '<span style="color: #4b2d73;">Warlock</span></h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="header-tagline">AI-powered CV tailoring for job applications</p>',
            unsafe_allow_html=True,
        )

    # How to use - collapsible
    with st.expander("How to use CV Warlock?"):
        st.markdown(
            """
            1. Paste your CV in the left panel
            2. Paste the job posting in the right panel (copy-pasted job title and description from LinkedIn)
            3. Click "Tailor My CV"
            4. Download your tailored CV!
            """
        )

    # CV Best Practices - collapsible
    with st.expander("CV Best Practices for 2026 (tech focused for now)"):
        st.markdown(
            """
CV should be organized into clear sections that tell your career story effectively. Below are the essential sections (in order), with notes on what each should contain:

### Length

Length expectations in Europe (2026): For senior-level applicants (CTO, Head of AI, Director roles), a two-page CV is commonly accepted and often expected. European CVs are essentially equivalent to US-style resumes and are typically kept concise: **"1 page is perfect, 2 are acceptable, 3 are rarely OK"**

### Design & Layout Best Practices in 2026

Modern CV design trends have shifted towards clarity and simplicity, especially in tech. In 2025–2026, the dominant trend is a clean, minimalist layout that emphasizes content over decoration.

#### Clean, Structured Layout (Single-Column & Clear Sections)

**Single-Column Format:** Use a one-column layout for optimal readability and ATS parsing. While multi-column templates became popular in the 2010s, today they're seen as less effective.

**Standard Sections & Headings:** Organize your CV into the familiar sections that tech recruiters expect. Typically, include: Contact Information, a strong Executive Summary/Profile, Skills/Competencies, Work Experience (reverse-chronological), Education, and optionally Certifications/Projects/Publications if relevant. Use conventional headings for these (e.g. "Professional Experience" rather than a creative label) – ATS scanners look for keywords like Experience, Education, Skills to categorize your info.

**Bullet Points & White Space:** Structure the details of your experience in bullet points (rather than long paragraphs) to improve skim-ability. Each bullet should highlight an accomplishment or responsibility, ideally quantified (e.g. "Implemented X, resulting in Y outcome"). Keep bullet points concise (1–2 lines each) and use 5-6 bullets for recent major roles, fewer for older ones.

#### Use of Color

In 2026, resume color schemes have trended back toward conservative palettes. The safest approach is a black (or dark gray) text on white background for the body of your CV. You can incorporate a limited accent color to give a modern touch – for example, a subtle navy or teal for headings or to underline your name at the top. If you do use an accent, keep it minimal and professional (avoid neon or overly bright tones) and ensure the CV is still clear if printed in grayscale.

#### Icons

**Icons and Symbols:** Resume icons (such as little symbols for phone, email, or skill graphs) became popular in graphic CV templates, but approach them with caution. Many hiring experts now advise avoiding icons entirely, or using them only in very limited ways, due to ATS and readability concerns.

Outdated Applicant Tracking Systems are still used by many big brands, and those cannot reliably interpret images or icons, so an icon might either be ignored or, worse, interpreted as a random character that garbles your CV's text.

**Graphics, Logos, and Photos:** For tech roles, avoid photos, graphics, or elaborate visuals on your CV. Including a headshot is common in some European countries, but the trend even in Europe is moving away from photos to reduce bias.

### ATS Compatibility Considerations

**Stick to Standard Formatting:** Use a traditional, simplified format that ATS software can easily parse. This means no text in headers or footers, no tables or multi-column text boxes, and no unusual fonts or encodings.

**File Format:** Submit in the format requested by the employer. Both PDF and DOCX are generally accepted by modern ATS systems. PDFs preserve your layout and have become largely ATS-safe (Jobscan's 2025 tests even found PDFs parse slightly more accurately on average).

**Keywords & Scannable Text:** Design choices should never hide the keywords your CV needs. For senior AI/ML roles (e.g. mentioning "LangChain, transformer models, cloud architecture"), make sure these terms appear as text in your CV (not buried in an image or graphic).

### Header & Contact Information

At the very top, include your name and updated contact details:

- Name (full name, bold or larger text).
- Email (professional address) and phone number (optional if you prefer calls).
- LinkedIn URL (since your LinkedIn can provide richer detail).
- Avoid adding a photo, age, or other personal data. They don't aid your candidacy and can introduce bias.

### Summary

A brief 3–5 sentence opening summary that highlights your unique value proposition as a leader. This should be tailored to the specific role/industry you target.

State your title or expertise (e.g. "AI Engineering Executive with 15+ years in ML innovation…").

Emphasize key achievements or specialties (for example, "led enterprise-wide AI transformations" or "built award-winning agentic AI platforms").

**Core Skills & Expertise:** A concise section (possibly a bullet list or inline list) highlighting your technical proficiencies and leadership skills most relevant to the role:

- **Technical Skills:** Include the emerging AI/ML technologies and tools you excel in. In 2026, AI fluency is crucial – mention things like "LLM orchestration (LangChain, Vector DBs)", "Deep Learning (PyTorch)", "Generative AI", and any domain-specific tools.
- **Leadership & Soft Skills:** Don't omit the human skills that are vital for executives. Highlight abilities like strategic planning, team leadership, cross-functional collaboration, communication, and change management.
- **Key Credentials:** If relevant, you can include certifications or noteworthy credentials here (e.g. "Ph.D. in Machine Learning", "MBA", "AWS Certified Solutions Architect").

### Professional Experience

This is the heart of your CV: detail your work history in reverse chronological order (most recent role first). For each role, focus on achievements and outcomes, not just duties:

**Heading:** Include your title, company, location, and years. If the company is not well-known, you can add a one-line description (e.g. "Series B SaaS startup" or "Fortune 500 e-commerce leader").

**Role Description:** 1–2 sentences (optional) summarizing your scope (e.g. "Led a 50-person data science and engineering organization with a $10M budget").

**Bulleted Achievements:** 2 to 4 bullet points per role highlighting your key accomplishments. Each bullet should demonstrate impact with metrics or concrete outcomes whenever possible. Use strong action verbs and be specific about technology and results. For example:

- "Grew data science team from 5 to 20 and implemented an ML platform that boosted model deployment speed 3×, enabling a 10x efficiency improvement in core operations"
- "Led cross-functional initiative to integrate GPT-4-powered agents into customer support, reducing response time by 60% and improving customer satisfaction by 25%"
- "Delivered a company-wide AI transformation (adopting automation and predictive analytics) that cut costs by $2M/year"

Each achievement bullet should answer: What did you do? How did you do it (tools/techniques)? and What was the result or value?

### Education & Certifications

Include your education and any important certifications. List degrees (Ph.D., Masters, Bachelors) with field of study, institution, and year. Mention certifications or courses that bolster your candidacy (e.g. "AWS Certified ML Specialty", "Certified Scrum Master", or executive leadership programs).

### Additional Sections (Optional)

Depending on your background, you may include one or more extra sections if they add real value for the target role. Ensure anything here is relevant and noteworthy, **quality over quantity.**

- **Key Achievements / Career Highlights:** A short bullet list (3–5 items) at the top of your CV that calls out your most impressive accomplishments across your career.
- **Projects or Portfolio:** Showcase major personal or open-source projects, prototypes, or research work outside of your formal employment.
- **Publications & Patents:** A brief section listing key publications or patents can establish your thought leadership and technical depth.
- **Awards & Honors:** Include notable recognitions.

Remember, focus on content that strengthens your fit for the specific role. Omit trivial or outdated items. Every section should add value or it can be removed.
            """
        )

    # Cover Letter Best Practices - collapsible
    with st.expander("Cover Letter Best Practices for 2026 (tech leadership focused)"):
        st.markdown(
            """
Cover letters have re-emerged as a critical tool for landing leadership roles. By 2025, 83% of hiring managers read cover letters (and nearly half read them before the resume). A personalized, results-driven cover letter distinguishes you from generic AI-generated text.

### Recommended Cover Letter Structure

**Header and Salutation:** Include your name, contact info, date, and address the letter to the hiring manager by name if possible. A personalized salutation creates a positive first impression.

**Opening Paragraph (Introduction):** Grab attention with a compelling hook—a top achievement or summary that directly relates to the job. State the role you're applying for and convey enthusiasm. For example: *"With 15+ years in IT leadership, I have driven 40% revenue growth through digital transformation…"*

**Body Paragraphs – Achievements and Value Alignment:** Showcase your most relevant accomplishments with metrics (e.g., "increased system uptime 30%" or "saved $1M in costs"). Mirror the job requirements and tie your expertise to their mission. Customization is crucial—72% of hiring managers prioritize tailored letters.

**Closing Paragraph:** Summarize your value proposition, express excitement, and include a call to action: *"I welcome the opportunity to discuss how my experience can help {Company} achieve Y."*

### Key Qualities to Highlight as a Tech Leader in 2026

- **Visionary Leadership:** Strategic foresight and technology roadmap experience
- **Adaptability & Agile Mindset:** Leading through change and uncertainty
- **AI Fluency & Innovation:** Comfort with AI, ML, and emerging technologies
- **Data-Driven Decision Making:** Analytical approach with measurable impact
- **Collaboration & Communication:** Bridging technology and business stakeholders
- **People Development:** Building high-performing teams and positive culture

By weaving these qualities with concrete examples, you paint a picture of a 2026-ready tech leader: someone with vision, adaptable, AI-fluent, data-driven, and capable of leading people through change.
            """
        )

    # Input columns
    col1, col2 = st.columns(2)

    with col1:
        raw_cv = render_cv_input()

    with col2:
        raw_job_spec = render_job_input()

    # Action button
    st.divider()

    _, col_btn2, _ = st.columns([1, 1, 1])
    with col_btn2:
        # Button always enabled (except during processing) - validation on click
        tailor_button = st.button(
            "Tailor My CV",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.processing,
        )

    # PHASE 1: Button clicked - validate, store params, start timer, rerun
    if tailor_button and not st.session_state.processing:
        # Validate both inputs are present
        if not raw_cv and not raw_job_spec:
            st.error("Please provide both a CV and a job specification.")
            return
        elif not raw_cv:
            st.error("Please provide your CV in the left panel.")
            return
        elif not raw_job_spec:
            st.error("Please provide a job specification in the right panel.")
            return

        # Get API key (from env or user input)
        effective_api_key = env_api_key or api_key

        if not effective_api_key:
            st.error(
                f"Please provide an API key for {provider.title()} in the sidebar or set it in .env.local"
            )
            return

        # Store all processing parameters in session state
        st.session_state.processing = True
        st.session_state.process_start_time = time.time()
        st.session_state.process_params = {
            "raw_cv": raw_cv,
            "raw_job_spec": raw_job_spec,
            "provider": provider,
            "api_key": effective_api_key,
            "use_cot": True,  # Always use high quality CoT mode
            "use_rlm": st.session_state.get("use_rlm", True),
            "lookback_years": st.session_state.get("lookback_years", 4),
            "assume_all_tech_skills": st.session_state.get("assume_all_tech_skills", True),
        }

        # Rerun to start Phase 2 (timer will be started there with known start time)
        st.rerun()

    # PHASE 2: Processing - timer already running, do the actual work
    if st.session_state.processing:
        params = st.session_state.process_params
        wall_start_time = st.session_state.process_start_time
        last_step = "Initializing"
        last_step_name = "start"

        # Track step completion for checklist
        completed_steps: set[str] = set()
        current_step_name: str | None = None
        current_step_desc: str | None = None
        failed_step_name: str | None = None

        try:
            # Import here to avoid loading dependencies until needed
            from cv_warlock.graph.workflow import run_cv_tailoring

            # Progress tracking with st.status
            status_label = "Tailoring your CV"
            if params["use_cot"]:
                status_label += " (CoT: reasoning + self-critique enabled)"
            status_label += "..."

            st.info(
                "⏳ Due to the thorough nature of the tailoring process, this can take a few minutes. "
                "Please be patient while we craft your optimized CV."
            )

            with st.status(status_label, expanded=True) as status:
                checklist_container = st.empty()
                timing_container = st.empty()

                def update_progress(step_name: str, description: str, _elapsed: float):
                    nonlocal last_step, last_step_name, current_step_name, current_step_desc
                    nonlocal completed_steps

                    # Mark previous step as completed (if it was a valid workflow step)
                    if current_step_name and current_step_name != step_name:
                        completed_steps.add(current_step_name)

                    last_step = description
                    last_step_name = step_name
                    current_step_name = step_name
                    current_step_desc = description

                    # Render the updated checklist
                    checklist_html = render_step_checklist(
                        completed_steps=completed_steps,
                        current_step=current_step_name,
                        failed_step=None,
                        current_description=current_step_desc,
                    )
                    checklist_container.markdown(
                        f'<div class="step-checklist">{checklist_html}</div>',
                        unsafe_allow_html=True,
                    )

                update_progress("graph_init", "Initializing graph...", 0)

                result = run_cv_tailoring(
                    raw_cv=params["raw_cv"],
                    raw_job_spec=params["raw_job_spec"],
                    provider=params["provider"],
                    api_key=params["api_key"],
                    progress_callback=update_progress,
                    assume_all_tech_skills=params["assume_all_tech_skills"],
                    use_cot=params["use_cot"],
                    use_rlm=params["use_rlm"],
                    lookback_years=params["lookback_years"],
                )

                # Calculate final times
                wall_clock_time = time.time() - wall_start_time
                api_time = result.get("total_generation_time", wall_clock_time)
                refinements = result.get("total_refinement_iterations", 0)

                # Check for workflow errors (stored in result["errors"])
                if result.get("errors"):
                    elapsed = time.time() - wall_start_time
                    failed_step_name = last_step_name

                    # Render checklist with failed step
                    checklist_html = render_step_checklist(
                        completed_steps=completed_steps,
                        current_step=None,
                        failed_step=failed_step_name,
                        current_description=None,
                    )
                    checklist_container.markdown(
                        f'<div class="step-checklist">{checklist_html}</div>',
                        unsafe_allow_html=True,
                    )

                    timing_container.markdown(
                        f'<div class="timing-display" style="background: #fff0f0; color: #cc0000;">'
                        f'<span class="realtime-timer">Failed after: {format_elapsed_time(elapsed)}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    status.update(
                        label=f"CV tailoring failed after {format_elapsed_time(elapsed)}",
                        state="error",
                    )

                    # Show detailed error information
                    st.divider()
                    st.subheader("Error Details")

                    for error_msg in result["errors"]:
                        error_info = parse_error_details(
                            error_msg, params["provider"], params["model"]
                        )
                        render_error_details(error_info, elapsed)

                    # Store failed result so user can see partial progress
                    st.session_state["result"] = None
                else:
                    # Success - mark all steps completed
                    all_step_names = {step[0] for step in WORKFLOW_STEPS}
                    checklist_html = render_step_checklist(
                        completed_steps=all_step_names,
                        current_step=None,
                        failed_step=None,
                        current_description=None,
                    )
                    checklist_container.markdown(
                        f'<div class="step-checklist">{checklist_html}</div>',
                        unsafe_allow_html=True,
                    )

                    timing_container.markdown(
                        f'<div class="timing-display">'
                        f'<span class="realtime-timer">Total time: {format_elapsed_time(wall_clock_time)}</span>'
                        f"<br>"
                        f'<span class="api-time">API processing: {format_elapsed_time(api_time)}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    completion_msg = (
                        f"CV tailored successfully! Total: {format_elapsed_time(wall_clock_time)}"
                    )
                    if params["use_cot"] and refinements > 0:
                        completion_msg += f" ({refinements} refinements)"
                    status.update(label=completion_msg, state="complete")

                    # Store result in session state
                    st.session_state["result"] = result

            # Reset processing state after status block completes (success or error)
            st.session_state.processing = False
            st.session_state.process_start_time = None

        except ImportError as e:
            elapsed = time.time() - wall_start_time
            error_info = {
                "category": "import",
                "title": "Missing Dependencies",
                "explanation": "Required packages are not installed.",
                "cause": f"Import error: {e}",
                "solution": [
                    "Run: uv sync --all-extras",
                    "Or: pip install -e .[dev]",
                    "Restart the Streamlit app",
                ],
                "technical": str(e),
            }
            render_error_details(error_info, elapsed)
            # Reset processing state
            st.session_state.processing = False
            st.session_state.process_start_time = None
            return

        except Exception as e:
            elapsed = time.time() - wall_start_time

            # Parse the exception for detailed feedback
            error_info = parse_error_details(str(e), params["provider"], params["model"])

            # Add context about where it failed
            st.divider()
            st.subheader("Error Details")
            st.markdown(f"**Failed during:** {last_step}")
            render_error_details(error_info, elapsed)

            # Show exception type for debugging
            with st.expander("Exception Type", expanded=False):
                st.code(f"{type(e).__name__}: {e}", language=None)

            # Reset processing state
            st.session_state.processing = False
            st.session_state.process_start_time = None
            return

    # Display results if available
    if "result" in st.session_state and st.session_state["result"] is not None:
        result = st.session_state["result"]
        render_result(result)

        # Show quality metrics if available (CoT mode)
        quality_scores = result.get("quality_scores")
        if quality_scores:
            with st.expander("Quality Assessment (CoT)", expanded=False):
                cols = st.columns(len(quality_scores))
                for i, (section, quality) in enumerate(quality_scores.items()):
                    with cols[i % len(cols)]:
                        section_name = section.replace("_", " ").title()
                        if quality == "excellent":
                            st.success(f"{section_name}: {quality}")
                        elif quality == "good":
                            st.info(f"{section_name}: {quality}")
                        elif quality == "needs_improvement":
                            st.warning(f"{section_name}: {quality}")
                        else:
                            st.error(f"{section_name}: {quality}")


if __name__ == "__main__":
    main()
