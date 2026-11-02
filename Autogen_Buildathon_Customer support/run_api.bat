@echo off
REM Start the FastAPI backend (Windows)
python -m uvicorn support_agent.api:app --host 127.0.0.1 --port 8000
