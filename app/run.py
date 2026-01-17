"""Simple launcher for the Streamlit web UI."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit web UI."""
    app_path = Path(__file__).parent / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    main()
