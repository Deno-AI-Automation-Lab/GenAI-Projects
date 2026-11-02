#!/usr/bin/env bash
# Start the FastAPI backend (Linux/macOS)
exec python -m uvicorn support_agent.api:app --host 127.0.0.1 --port 8000
