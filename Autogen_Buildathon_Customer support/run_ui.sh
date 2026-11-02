#!/usr/bin/env bash
# Start the Streamlit frontend (Linux/macOS)
exec python -m streamlit run support_agent/ui.py --server.port 8501
