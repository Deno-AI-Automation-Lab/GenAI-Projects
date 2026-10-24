"""Frontend entry point.  Run from the project root:  streamlit run frontend/app.py

The RAG logic lives in backend/hybrid_rag; this adds backend/ to the import
path (pathlib, so it works the same on Windows and Linux).
"""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from ui import main  # noqa: E402  (frontend/ui.py)

main()
