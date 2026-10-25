"""Central configuration. Secrets come from the .env file (never hard-coded)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Quiet CrewAI's anonymous telemetry (optional)
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Models
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Paths
DATA_DIR = BASE_DIR / "data"
PDF_PATH = Path(os.getenv(
    "PDF_PATH",
    DATA_DIR / "FiberNet_Broadband_Network_Troubleshooting_Handbook_20_Pages.pdf",
))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", BASE_DIR / "vectorstore" / "fibernet_faiss"))

# Chunking / retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Web search (Web Research Agent) for questions the handbook doesn't cover
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")   # optional; DuckDuckGo is used if empty
WEB_MAX_RESULTS = int(os.getenv("WEB_MAX_RESULTS", "5"))
WEB_REGION = os.getenv("WEB_REGION", "in-en")      # DuckDuckGo region (India / English)

# Google Sheets query log
GSHEET_ENABLED = os.getenv("GSHEET_ENABLED", "true").lower() == "true"
GSHEET_ID = os.getenv("GSHEET_ID", "")                       # the long ID in the sheet URL
GSHEET_WORKSHEET = os.getenv("GSHEET_WORKSHEET", "Query Log")
_sa = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "")
GOOGLE_SERVICE_ACCOUNT_FILE = str((BASE_DIR / _sa) if _sa and not Path(_sa).is_absolute() else _sa)


def require_api_key() -> None:
    if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to the .env file in the project folder "
            "(see .env.example)."
        )
