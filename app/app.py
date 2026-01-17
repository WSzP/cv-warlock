"""Streamlit web UI for CV Warlock."""

import os
import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env.local
from dotenv import load_dotenv

load_dotenv(project_root / ".env.local")

import streamlit as st

from components.cv_input import render_cv_input
from components.job_input import render_job_input
from components.result_display import render_result
from utils.styles import apply_custom_styles


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
    if any(x in error_lower for x in ["unauthorized", "401", "invalid api key", "authentication",
                                       "invalid_api_key", "invalid x-api-key"]):
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
    if any(x in error_lower for x in ["rate limit", "rate_limit", "429", "too many requests",
                                       "quota", "exceeded"]):
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
    if any(x in error_lower for x in ["model not found", "invalid model", "model_not_found",
                                       "does not exist", "not available"]):
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
    if any(x in error_lower for x in ["context length", "token limit", "too long", "maximum context"]):
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
    if any(x in error_lower for x in ["extraction failed", "parsing", "parse error", "invalid json",
                                       "failed to extract"]):
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
    if any(x in error_lower for x in ["500", "502", "503", "504", "server error", "internal error"]):
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


# Page configuration
st.set_page_config(
    page_title="CV Warlock",
    page_icon=":magic_wand:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Poppins font styling
apply_custom_styles()

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
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
        st.header("Settings")

        provider = st.selectbox(
            "LLM Provider",
            options=["anthropic", "openai", "google"],
            index=0,
            help="Choose the AI provider for CV tailoring",
        )

        # Model options per provider with latest versions
        if provider == "openai":
            model_options = [
                "gpt-5.2",           # Latest flagship
                "gpt-5.2-instant",   # Fast version
                "gpt-5-mini",        # Cost-efficient
                "gpt-4o",
            ]
        elif provider == "google":
            model_options = [
                "gemini-3-flash-preview",  # Fast + capable (recommended)
                "gemini-3-pro-preview",    # Most capable
            ]
        else:  # anthropic
            model_options = [
                "claude-sonnet-4-5-20250929",  # Best balance (recommended)
                "claude-haiku-4-5-20251001",   # Fastest, cost-efficient
                "claude-opus-4-5-20251101",    # Most capable
            ]

        model = st.selectbox(
            "Model",
            options=model_options,
            index=0,
            help="Select the specific model to use",
        )

        # Show runtime warning for slower models
        if "opus" in model.lower():
            st.info(
                "**Opus runtime:** 3-5 minutes expected. "
                "Other models typically complete in under 1 minute.",
                icon="⏱️"
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

        st.divider()

        st.markdown(
            """
            ### How it works
            1. Paste your CV in the left panel
            2. Paste the job posting in the right panel
            3. Click "Tailor My CV"
            4. Download your tailored CV!
            """
        )

    # Main content
    st.markdown('<p class="main-header">CV Warlock</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-powered CV tailoring for job applications</p>',
        unsafe_allow_html=True,
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
            "model": model,
            "api_key": effective_api_key,
            "use_cot": True,  # Always use high quality CoT mode
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

        try:
            # Import here to avoid loading dependencies until needed
            from cv_warlock.graph.workflow import run_cv_tailoring

            # Progress tracking with st.status
            status_label = "Tailoring your CV"
            if params["use_cot"]:
                status_label += " (CoT: reasoning + self-critique enabled)"
            status_label += "..."

            with st.status(status_label, expanded=True) as status:
                progress_container = st.empty()
                timing_container = st.empty()

                def update_progress(_step_name: str, description: str, _elapsed: float):
                    nonlocal last_step
                    last_step = description
                    progress_container.markdown(f"**{description}**")

                update_progress("start", "Initializing...", 0)

                result = run_cv_tailoring(
                    raw_cv=params["raw_cv"],
                    raw_job_spec=params["raw_job_spec"],
                    provider=params["provider"],
                    model=params["model"],
                    api_key=params["api_key"],
                    progress_callback=update_progress,
                    assume_all_tech_skills=params["assume_all_tech_skills"],
                    use_cot=params["use_cot"],
                    lookback_years=params["lookback_years"],
                )

                # Calculate final times
                wall_clock_time = time.time() - wall_start_time
                api_time = result.get("total_generation_time", wall_clock_time)
                refinements = result.get("total_refinement_iterations", 0)

                # Check for workflow errors (stored in result["errors"])
                if result.get("errors"):
                    elapsed = time.time() - wall_start_time
                    timing_container.markdown(
                        f'<div class="timing-display" style="background: #fff0f0; color: #cc0000;">'
                        f'<span class="realtime-timer">Failed after: {format_elapsed_time(elapsed)}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    status.update(
                        label=f"CV tailoring failed after {format_elapsed_time(elapsed)}",
                        state="error",
                    )
                    progress_container.markdown(f"**Failed during:** {last_step}")

                    # Show detailed error information
                    st.divider()
                    st.subheader("Error Details")

                    for error_msg in result["errors"]:
                        error_info = parse_error_details(error_msg, params["provider"], params["model"])
                        render_error_details(error_info, elapsed)

                    # Store failed result so user can see partial progress
                    st.session_state["result"] = None
                else:
                    # Success
                    timing_container.markdown(
                        f'<div class="timing-display">'
                        f'<span class="realtime-timer">Total time: {format_elapsed_time(wall_clock_time)}</span>'
                        f'<br>'
                        f'<span class="api-time">API processing: {format_elapsed_time(api_time)}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    completion_msg = f"CV tailored successfully! Total: {format_elapsed_time(wall_clock_time)}"
                    if params["use_cot"] and refinements > 0:
                        completion_msg += f" ({refinements} refinements)"
                    status.update(label=completion_msg, state="complete")
                    progress_container.markdown("**Done!**")

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
