"""Root entrypoint proxy for Streamlit Cloud deployment of SoundVision AI."""
from pathlib import Path
import runpy

app_target = Path(__file__).resolve().parent / "app" / "app.py"
runpy.run_path(str(app_target), run_name="__main__")
