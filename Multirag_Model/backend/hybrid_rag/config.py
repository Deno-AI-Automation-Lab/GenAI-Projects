"""Settings, prompts and shared constants.

Everything configurable lives here so the other modules stay free of magic
values. Secrets are read from `.env` (never hard-coded).
"""
from __future__ import annotations

import os
import re
import ssl
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths (pathlib only -> identical behaviour on Windows and Linux)
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent             # backend/
PROJECT_DIR = BACKEND_DIR.parent             # repo root: .env, fonts/, eval/, data/
load_dotenv(PROJECT_DIR / ".env")  # secrets: OPENAI_API_KEY, NEO4J_*, LANGSMITH_*

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_DIR / "data"))
INDEX_DIR = DATA_DIR / "index"          # persisted FAISS + chunk JSON
CACHE_DIR = DATA_DIR / "cache"          # OCR / summary cache keyed by content hash
UPLOAD_DIR = DATA_DIR / "uploads"
FONTS_DIR = PROJECT_DIR / "fonts"
TAMIL_FONT = FONTS_DIR / "NotoSansTamil-Regular.ttf"   # bundled fallback font
EVAL_SET = PROJECT_DIR / "eval" / "eval_set.json"
for _d in (DATA_DIR, INDEX_DIR, CACHE_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Multilingual token pattern - defined ONCE, reused by BM25, entity matching,
# guardrails and language detection. Covers Latin letters/digits, Tamil
# (U+0B80-U+0BFF) and Devanagari (U+0900-U+097F). Add ranges here only.
# ---------------------------------------------------------------------------
TAMIL_RANGE = "\u0B80-\u0BFF"
DEVANAGARI_RANGE = "\u0900-\u097F"
TOKEN_PATTERN = re.compile("[A-Za-z0-9" + TAMIL_RANGE + DEVANAGARI_RANGE + "]+", re.UNICODE)
TAMIL_CHAR = re.compile("[" + TAMIL_RANGE + "]")


def tokenize(text: str) -> list[str]:
    """Lower-cased tokens using the shared multilingual pattern."""
    return [t.lower() for t in TOKEN_PATTERN.findall(text or "")]


# ---------------------------------------------------------------------------
# SSL: never verify=False. Use the OS trust store (truststore) and optionally a
# bundled CA/intermediate file for sites with incomplete certificate chains.
# ---------------------------------------------------------------------------
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - truststore unavailable: keep default certifi
    pass
CA_BUNDLE = os.getenv("CA_BUNDLE")  # optional path to extra CA bundle (.pem)


def requests_verify() -> bool | str:
    """Value for requests' `verify=`: a CA bundle path or True. Never False."""
    if CA_BUNDLE and Path(CA_BUNDLE).exists():
        return str(Path(CA_BUNDLE))
    return True


def ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if CA_BUNDLE and Path(CA_BUNDLE).exists():
        ctx.load_verify_locations(cafile=str(Path(CA_BUNDLE)))
    return ctx


# ---------------------------------------------------------------------------
# Runtime settings
# ---------------------------------------------------------------------------
RETRIEVAL_MODES = ("vector", "bm25", "hybrid", "hybrid+graph")


@dataclass
class Settings:
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    vision_model: str = os.getenv("VISION_MODEL", "gpt-4o")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid+graph")
    top_k: int = int(os.getenv("TOP_K", "5"))
    per_retriever_k: int = int(os.getenv("PER_RETRIEVER_K", "10"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))           # RRF damping constant
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    neighbour_window: int = int(os.getenv("NEIGHBOUR_WINDOW", "1"))
    history_turns: int = int(os.getenv("HISTORY_TURNS", "6"))
    max_question_chars: int = int(os.getenv("MAX_QUESTION_CHARS", "1000"))
    graph_max_rows: int = int(os.getenv("GRAPH_MAX_ROWS", "25"))
    min_scanned_chars: int = int(os.getenv("MIN_SCANNED_CHARS", "40"))  # below -> OCR
    category_field: str = os.getenv("CATEGORY_FIELD", "department")
    # crawler
    source_website: str = os.getenv("SOURCE_WEBSITE", "")
    crawl_max_pages: int = int(os.getenv("CRAWL_MAX_PAGES", "30"))
    crawl_delay: float = float(os.getenv("CRAWL_DELAY", "1.0"))
    user_agent: str = os.getenv("CRAWL_USER_AGENT", "HybridRAGBot/1.0")
    # neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_user: str = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"RETRIEVAL_MODE must be one of {RETRIEVAL_MODES}")

    @property
    def graph_enabled(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_password)


def get_settings(**overrides) -> Settings:
    """Fresh settings object; keyword overrides are handy in tests/scripts."""
    return Settings(**overrides)


# LangSmith: tracing is switched on purely through env vars that LangChain reads.
LANGSMITH_ENABLED = bool(os.getenv("LANGSMITH_API_KEY")) and os.getenv(
    "LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "true")
).lower() == "true"
if LANGSMITH_ENABLED:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "hybrid-rag-pdf")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
NO_ANSWER = "I couldn't find this in the uploaded documents."

ANSWER_SYSTEM_PROMPT = f"""You are a document assistant. Answer ONLY from the CONTEXT below,
which comes from the user's indexed PDF documents (vector store and knowledge graph).
Rules:
- Never use outside knowledge, never browse or search the web, never guess.
- If the context does not contain the answer, reply exactly: "{NO_ANSWER}"
- Cite sources inline as [S1], [S2] ... matching the context labels; graph facts as [G].
- Answer in the language of the question. Be concise and factual.

CONTEXT:
{{context}}"""

PLANNER_PROMPT = """You plan retrieval for a document Q&A system.
Given the chat history and the latest question:
1. Rewrite the question into a standalone question (resolve pronouns/follow-ups).
2. List named entities (schemes, programmes, organisations, products, people) mentioned.
3. If the question clearly targets one {category_field}, set category_filter to it, else null.
   Known {category_field} values: {categories}
4. needs_graph = true when the question is about relationships between entities
   (who issues what, what belongs to what, eligibility links, comparisons).
5. keywords: 3-8 important search terms (keep original language)."""

GRAPH_EXTRACT_PROMPT = """Extract a knowledge graph from the text.
Return entities (name, type) and relations (source, relation, target).
Use short canonical names, UPPER_SNAKE_CASE relation types (e.g. ISSUED_BY,
ELIGIBLE_FOR, PROVIDES, PART_OF, HAS_AMOUNT, HAS_DEADLINE). Max 15 relations.
Only extract facts stated in the text."""

CYPHER_PROMPT = """Write ONE read-only Cypher query for Neo4j that answers the question.
Schema:
  (:Entity {{name, type}})-[:REL {{type, chunk_id}}]->(:Entity)
  (:Chunk {{chunk_id, text}})-[:MENTIONS]->(:Entity)
Rules: only MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / LIMIT / UNWIND.
Match names case-insensitively with toLower(e.name) CONTAINS toLower('...').
Return columns: source, relation, target, chunk_id. LIMIT {limit}.
Entities from the question: {entities}
Question: {question}"""

KEY_FACTS_PROMPT = """The following document text is not in English.
Write an English "KEY FACTS" summary as bullet points covering, where present:
eligibility, benefits, amounts, dates/deadlines, issuing authority, how to apply.
Do not invent facts.

TEXT:
{text}"""

OCR_PROMPT = ("Transcribe all text on this scanned page exactly, in its original "
              "language. Render tables as Markdown tables. Output text only.")

JUDGE_PROMPT = """You are grading a RAG system answer. Score each 1-5:
- faithfulness: is every claim supported by the CONTEXT?
- relevance: does it answer the QUESTION?
- correctness: does it agree with the REFERENCE answer (5 if no reference)?
QUESTION: {question}
REFERENCE: {reference}
CONTEXT: {context}
ANSWER: {answer}"""
