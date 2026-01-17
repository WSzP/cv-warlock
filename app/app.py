"""Streamlit web UI for CV Warlock."""

import streamlit as st

from app.components.cv_input import render_cv_input
from app.components.job_input import render_job_input
from app.components.result_display import render_result

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
            options=["openai", "anthropic"],
            index=0,
            help="Choose the AI provider for CV tailoring",
        )

        if provider == "openai":
            default_model = "gpt-4o"
            model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
        else:
            default_model = "claude-opus-4-5-20251101"
            model_options = ["claude-opus-4-5-20251101", "claude-sonnet-4-20250514"]

        model = st.selectbox(
            "Model",
            options=model_options,
            index=0,
            help="Select the specific model to use",
        )

        api_key = st.text_input(
            f"{provider.title()} API Key",
            type="password",
            help="Enter your API key (or set it in .env file)",
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

        # Check for API key
        effective_api_key = api_key or None
        if not effective_api_key:
            # Try to load from environment
            import os

            if provider == "openai":
                effective_api_key = os.getenv("OPENAI_API_KEY")
            else:
                effective_api_key = os.getenv("ANTHROPIC_API_KEY")

        if not effective_api_key:
            st.error(
                f"Please provide an API key for {provider.title()} in the sidebar or set it in your .env file."
            )
            return

        # Run tailoring
        with st.spinner("Tailoring your CV... This may take a minute."):
            try:
                # Import here to avoid loading dependencies until needed
                from cv_warlock.graph.workflow import run_cv_tailoring

                result = run_cv_tailoring(
                    raw_cv=raw_cv,
                    raw_job_spec=raw_job_spec,
                    provider=provider,
                    model=model,
                    api_key=effective_api_key,
                )

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
