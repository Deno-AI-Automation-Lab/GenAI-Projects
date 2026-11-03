# AutoGen Buildathon Customer Support - Dockerized

The project is separated into three deployment areas:

```text
.
├── frontend/                # Streamlit UI (port 8501)
│   ├── app.py
│   ├── client.py
│   ├── assets/
│   ├── Dockerfile
│   └── requirements.txt
├── backend/                 # FastAPI + AutoGen + LangChain RAG (port 8000)
│   ├── support_agent/
│   ├── data/                # product knowledge-base documents
│   ├── secrets/             # optional Google service-account JSON (not included)
│   ├── Dockerfile
│   └── requirements.txt
├── database/                # persistent application/graph data
│   ├── storage/             # SQLite, CSV, FAISS/BM25 indexes, caches
│   └── neo4j/               # Neo4j runtime data
├── docker-compose.yml
└── .env.example
```

## Start

1. Copy the example environment file:
   - Linux/macOS: `cp .env.example .env`
   - Windows PowerShell: `Copy-Item .env.example .env`
2. Put your OpenAI key and other required values into `.env`.
3. Optional Google Sheets integration: place the service-account file at
   `backend/secrets/google-service-account.json`.
4. Build and start:

```bash
docker compose up -d --build
```

Open:
- Streamlit UI: http://localhost:8501
- FastAPI: http://localhost:8000
- FastAPI docs: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474

Check containers:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs -f backend
docker compose logs -f frontend
```

Stop:

```bash
docker compose down
```

Persistent data stays inside `database/` after containers are removed.

## Rebuild RAG indexes

The existing indexes from the supplied project are retained under `database/storage/indexes`.
To rebuild from the knowledge-base documents:

```bash
docker compose exec backend python -m support_agent.ingest
```

## Security

The original `.env` and Google service-account JSON were intentionally excluded from this converted package. Never commit secrets to Git or bake them into Docker images.
