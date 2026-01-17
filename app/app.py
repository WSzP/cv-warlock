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


# Page configuration
st.set_page_config(
    page_title="CV Warlock",
    page_icon=":magic_wand:",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Main Streamlit application."""
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

        # Temperature slider
        default_temp = float(os.getenv("CV_WARLOCK_TEMPERATURE", "0.3"))
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=default_temp,
            step=0.1,
            help="Controls randomness. Lower = more focused, higher = more creative",
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

        # Generation quality settings
        st.subheader("Generation Quality")

        use_cot = st.checkbox(
            "**High Quality** with Chain-of-Thought (CoT) Reasoning",
            value=True,
            help="Enable multi-step reasoning for higher quality output. "
            "Each section goes through: Reason → Generate → Critique → Refine. "
            "Produces better results but takes 3-4x longer.",
            key="use_cot",
        )

        if use_cot:
            st.info(
                "CoT enabled: Generation will be slower but produces "
                "significantly higher quality tailored CVs."
            )
        else:
            st.warning(
                "CoT disabled: Faster generation but lower quality. "
                "Use for quick iterations."
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
                dashboard_url = f"https://eu.smith.langchain.com/o/default/projects/p/{project}"
            else:
                dashboard_url = f"https://smith.langchain.com/o/default/projects/p/{project}"
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

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        tailor_button = st.button(
            "Tailor My CV",
            type="primary",
            use_container_width=True,
            disabled=not (raw_cv and raw_job_spec),
        )

    # Process and display results
    if tailor_button:
        if not raw_cv or not raw_job_spec:
            st.error("Please provide both a CV and a job specification.")
            return

        # Get API key (from env or user input)
        effective_api_key = env_api_key or api_key

        if not effective_api_key:
            st.error(
                f"Please provide an API key for {provider.title()} in the sidebar or set it in .env.local"
            )
            return

        # Run tailoring with detailed progress and timing
        try:
            # Import here to avoid loading dependencies until needed
            from cv_warlock.graph.workflow import run_cv_tailoring

            # Get settings from session state
            assume_all_tech_skills = st.session_state.get("assume_all_tech_skills", True)
            use_cot_setting = st.session_state.get("use_cot", True)

            # Progress tracking with st.status
            status_label = "Tailoring your CV"
            if use_cot_setting:
                status_label += " (CoT: reasoning + self-critique enabled)"
            status_label += "..."

            with st.status(status_label, expanded=True) as status:
                progress_container = st.empty()
                timing_container = st.empty()
                start_time = time.time()

                # Show initial timing
                timing_container.markdown(
                    f'<div class="timing-display">Elapsed: 0.0s</div>',
                    unsafe_allow_html=True,
                )

                def update_progress(step_name: str, description: str, elapsed: float):
                    # Update both progress description and elapsed time
                    progress_container.markdown(f"**{description}**")
                    # Use wall-clock time for more accurate display
                    actual_elapsed = time.time() - start_time
                    timing_container.markdown(
                        f'<div class="timing-display">Elapsed: {format_elapsed_time(actual_elapsed)}</div>',
                        unsafe_allow_html=True,
                    )

                update_progress("start", "Initializing...", 0)

                result = run_cv_tailoring(
                    raw_cv=raw_cv,
                    raw_job_spec=raw_job_spec,
                    provider=provider,
                    model=model,
                    api_key=effective_api_key,
                    progress_callback=update_progress,
                    assume_all_tech_skills=assume_all_tech_skills,
                    use_cot=use_cot_setting,
                    temperature=temperature,
                )

                # Final timing
                total_time = result.get("total_generation_time", time.time() - start_time)
                refinements = result.get("total_refinement_iterations", 0)

                if result.get("errors"):
                    status.update(label="CV tailoring failed", state="error")
                else:
                    completion_msg = f"CV tailored successfully! Total time: {format_elapsed_time(total_time)}"
                    if use_cot_setting and refinements > 0:
                        completion_msg += f" ({refinements} refinement iterations)"
                    status.update(label=completion_msg, state="complete")
                    progress_container.markdown("**Done!**")
                    # Show final timing
                    timing_container.markdown(
                        f'<div class="timing-display">Total time: {format_elapsed_time(total_time)}</div>',
                        unsafe_allow_html=True,
                    )

            # Store result in session state
            st.session_state["result"] = result

        except Exception as e:
            st.error(f"An error occurred: {e}")
            return

    # Display results if available
    if "result" in st.session_state:
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
