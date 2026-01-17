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
POPPINS_CSS = """
<style>

    /* Apply Poppins to all elements */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }

    /* Main title */
    .stApp h1 {
        font-weight: 700;
    }

    /* Subheaders */
    .stApp h2, .stApp h3 {
        font-weight: 600;
    }

    /* Text areas and inputs */
    .stTextArea textarea, .stTextInput input {
        font-family: 'Poppins', sans-serif;
    }

    /* Buttons */
    .stButton button {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Download buttons */
    .stDownloadButton button {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
    }

    /* Markdown content */
    .stMarkdown {
        font-family: 'Poppins', sans-serif;
    }

    /* Code blocks should keep monospace */
    code, pre, .stCode {
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
