"""Custom styles for CV Warlock Streamlit UI.

Applies Poppins font family throughout the application for a consistent,
modern look that matches the PDF output.
"""

import streamlit as st

# Poppins font - preconnect for faster loading, then load all weights
POPPINS_PRECONNECT = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" rel="stylesheet">
"""

# Poppins font CSS - applies the font throughout the UI
# Uses !important to override Streamlit's default "Source Sans" font
POPPINS_CSS = """
<style>
    /* Global override - force Poppins everywhere */
    *, *::before, *::after {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Apply Poppins to all elements */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Main title */
    .stApp h1 {
        font-weight: 700 !important;
    }

    /* Subheaders */
    .stApp h2, .stApp h3 {
        font-weight: 600 !important;
    }

    /* Text areas and inputs */
    .stTextArea textarea, .stTextInput input {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Buttons */
    .stButton button, button {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"], [data-testid="stSidebar"] * {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Metrics */
    [data-testid="stMetric"], [data-testid="stMetric"] * {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Download buttons */
    .stDownloadButton button {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
    }

    /* Markdown content */
    .stMarkdown, .stMarkdown * {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Selectbox and other inputs */
    [data-baseweb="select"], [data-baseweb="input"] {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Code blocks should keep monospace - this overrides the global rule */
    code, pre, .stCode, code *, pre * {
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
    }
</style>
"""


def apply_custom_styles() -> None:
    """Apply custom Poppins font styling to the Streamlit app.

    Call this at the beginning of your app.py to apply consistent styling.
    Uses preconnect for faster font loading from Google Fonts.
    """
    # Preconnect to Google Fonts for faster loading
    st.markdown(POPPINS_PRECONNECT, unsafe_allow_html=True)
    # Apply font styling
    st.markdown(POPPINS_CSS, unsafe_allow_html=True)
