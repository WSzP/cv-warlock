"""Simple launcher for the Streamlit web UI."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit web UI."""
    app_path = Path(__file__).parent / "app.py"
    stop_hint = "⌃C (Control+C)" if sys.platform == "darwin" else "Ctrl+C"
    print(f"Press {stop_hint} to stop the server")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        pass


if __name__ == "__main__":
    main()
