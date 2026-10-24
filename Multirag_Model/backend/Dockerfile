# syntax=docker/dockerfile:1
# Backend image: FastAPI + RAG core.  Build from the PROJECT ROOT:
#   docker build -f backend/Dockerfile -t multirag-backend .
# Secrets are NOT baked in - pass them at run time with --env-file .env
FROM python:3.12-slim-bookworm

# "true" bundles Chromium for crawling JavaScript-heavy websites (+~400 MB)
ARG INSTALL_PLAYWRIGHT=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PYTHONPATH=/app/backend \
    DATA_DIR=/app/data \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# ca-certificates: TLS verification (truststore uses the OS store); libgomp1: FAISS
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt pytest==9.1.1 httpx==0.28.1 \
 && if [ "$INSTALL_PLAYWRIGHT" = "true" ]; then python -m playwright install --with-deps chromium; fi

# Same layout as the repo, so config.PROJECT_DIR resolves to /app
COPY backend/ backend/
COPY fonts/ fonts/
COPY eval/ eval/
COPY samples/ samples/
COPY scripts/ scripts/
COPY tests/ tests/
COPY pytest.ini ./

RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /app/data/index /app/data/cache /app/data/uploads \
 && chown -R appuser:appuser /app
USER appuser
VOLUME ["/app/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "api:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "8000"]
