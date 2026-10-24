# syntax=docker/dockerfile:1
# Frontend image: Streamlit UI.  Build from the PROJECT ROOT:
#   docker build -f frontend/Dockerfile -t multirag-frontend .
# The UI runs the RAG pipeline in-process (store kept in st.session_state),
# so it also contains the backend's core package (backend/hybrid_rag).
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PYTHONPATH=/app/backend \
    DATA_DIR=/app/data

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
COPY frontend/requirements.txt frontend/requirements.txt
RUN pip install -r frontend/requirements.txt

COPY backend/hybrid_rag/ backend/hybrid_rag/
COPY frontend/ frontend/
COPY fonts/ fonts/
COPY eval/ eval/

RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /app/data/index /app/data/cache /app/data/uploads \
 && chown -R appuser:appuser /app
USER appuser
VOLUME ["/app/data"]

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false", "--server.maxUploadSize=200"]
