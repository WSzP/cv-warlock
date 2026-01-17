"""Streamlit web UI for CV Warlock."""

import os
import sys
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
    else:
        return os.getenv("ANTHROPIC_API_KEY")

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
            options=["anthropic", "openai"],
            index=0,
            help="Choose the AI provider for CV tailoring",
        )

        if provider == "openai":
            model_options = ["gpt-5.2", "gpt-4o", "gpt-4o-mini"]
        else:
            model_options = ["claude-opus-4-5-20251101", "claude-sonnet-4-20250514"]

        model = st.selectbox(
            "Model",
            options=model_options,
            index=0,
            help="Select the specific model to use",
        )

        # Check if API key exists in environment
        env_api_key = get_env_api_key(provider)

        # Only show API key input if not in environment
        if env_api_key:
            st.success(f"{provider.title()} API key loaded from .env.local")
            api_key = None  # Will use env key
        else:
            api_key = st.text_input(
                f"{provider.title()} API Key",
                type="password",
                help="Enter your API key (or set it in .env.local)",
            )

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

        # Run tailoring with detailed progress
        try:
            # Import here to avoid loading dependencies until needed
            from cv_warlock.graph.workflow import run_cv_tailoring

            # Progress tracking with st.status
            with st.status("Tailoring your CV...", expanded=True) as status:
                progress_container = st.empty()
                current_step = {"name": "", "desc": "Starting..."}

                def update_progress(step_name: str, description: str):
                    current_step["name"] = step_name
                    current_step["desc"] = description
                    progress_container.markdown(f"**{description}**")

                update_progress("start", "Initializing...")

                result = run_cv_tailoring(
                    raw_cv=raw_cv,
                    raw_job_spec=raw_job_spec,
                    provider=provider,
                    model=model,
                    api_key=effective_api_key,
                    progress_callback=update_progress,
                )

                # Update status on completion
                if result.get("errors"):
                    status.update(label="CV tailoring failed", state="error")
                else:
                    status.update(label="CV tailored successfully!", state="complete")
                    progress_container.markdown("**Done!**")

            # Store result in session state
            st.session_state["result"] = result

        except Exception as e:
            st.error(f"An error occurred: {e}")
            return

    # Display results if available
    if "result" in st.session_state:
        render_result(st.session_state["result"])


if __name__ == "__main__":
    main()
