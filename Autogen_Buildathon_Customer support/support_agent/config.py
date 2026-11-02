"""Central settings, prompts and shared constants.

Everything configurable lives here and is read from the environment / `.env`
(never hard-coded secrets). Other modules import `get_settings()` and the
constants below instead of re-defining them.
"""
from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Paths (pathlib only -> works on Windows and Linux)
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
FONTS_DIR: Path = ASSETS_DIR / "fonts"
TAMIL_FONT_PATH: Path = FONTS_DIR / "NotoSansTamil-Regular.ttf"

# --------------------------------------------------------------------------- #
# Multilingual token pattern – defined ONCE and reused by BM25, entity matching,
# guardrails and evaluation. Covers Latin (incl. accented), Devanagari (Hindi),
# Tamil (஀-௿), Telugu, Kannada, Malayalam, Arabic, CJK and digits.
# --------------------------------------------------------------------------- #
TOKEN_PATTERN: str = (
    r"[A-Za-z0-9À-ɏ"      # Latin + accented Latin (fr/es/de/pt)
    r"ऀ-ॿ"                 # Devanagari (Hindi/Marathi)
    r"஀-௿"                 # Tamil
    r"ఀ-౿"                 # Telugu
    r"ಀ-೿"                 # Kannada
    r"ഀ-ൿ"                 # Malayalam
    r"؀-ۿ"                 # Arabic
    r"一-鿿"                 # CJK unified ideographs
    r"]+"
)
TOKEN_RE: re.Pattern[str] = re.compile(TOKEN_PATTERN)

# Order IDs like ORD-12345, #A1B2C3, AB123456 (used for mail intake + guardrails)
ORDER_ID_RE: re.Pattern[str] = re.compile(r"\b(?:ORD[-_ ]?\d{4,}|#?(?!TKT)[A-Z]{2,4}-?\d{5,10})\b", re.I)


def tokenize(text: str) -> list[str]:
    """Lower-cased multilingual tokens using the shared TOKEN_PATTERN."""
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #
DOC_TYPES: tuple[str, ...] = ("shipping_policy", "refund_policy", "account_and_login", "product_faq")


class RetrievalMode(str, Enum):
    """Selectable retrieval strategies so approaches can be compared in evaluation."""

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"              # vector + bm25 + entity-summary + exact-entity (RRF)
    HYBRID_GRAPH = "hybrid+graph"  # hybrid + Neo4j traversal (RRF)


CATEGORIES: tuple[str, ...] = (
    "shipping", "refund", "account_login", "product_usage",
    "technical_issue", "payment", "warranty", "other",
)
# Maps a ticket category to the document type that should be favoured.
CATEGORY_TO_DOCTYPE: dict[str, str] = {
    "shipping": "shipping_policy",
    "refund": "refund_policy",
    "payment": "refund_policy",
    "account_login": "account_and_login",
    "product_usage": "product_faq",
    "technical_issue": "product_faq",
    "warranty": "product_faq",
}
URGENCY_SCORE: dict[str, int] = {"Low": 1, "Medium": 2, "High": 3}

# Escalation keywords (deterministic layer on top of LLM signals).
LEGAL_FRAUD_PAYMENT_TERMS: tuple[str, ...] = (
    "lawyer", "legal action", "sue", "court", "consumer forum", "lawsuit", "police",
    "fraud", "scam", "stolen", "unauthorized", "unauthorised", "chargeback",
    "double charged", "charged twice", "payment dispute", "dispute the charge",
    "வழக்கு", "மோசடி",                         # Tamil: case/lawsuit, fraud
    "धोखाधड़ी", "कानूनी",                      # Hindi: fraud, legal
)


class Settings(BaseSettings):
    """All runtime settings. Values come from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # OpenAI (LangChain ChatOpenAI + OpenAIEmbeddings, AutoGen model client)
    openai_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENAI_API_KEY")
    chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    agent_model: str = Field(default="gpt-4o-mini", alias="OPENAI_AGENT_MODEL")
    judge_model: str = Field(default="gpt-4o-mini", alias="OPENAI_JUDGE_MODEL")
    vision_model: str = Field(default="gpt-4o", alias="OPENAI_VISION_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    llm_timeout_s: int = Field(default=60, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: SecretStr = Field(default=SecretStr(""), alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    # Data locations (relative paths resolve against the project root)
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="DATA_DIR")
    index_dir: Path = Field(default=PROJECT_ROOT / "storage" / "indexes", alias="INDEX_DIR")
    cache_dir: Path = Field(default=PROJECT_ROOT / "storage" / "cache", alias="CACHE_DIR")
    db_path: Path = Field(default=PROJECT_ROOT / "storage" / "support_memory.db", alias="MEMORY_DB_PATH")
    log_csv_path: Path = Field(default=PROJECT_ROOT / "storage" / "ticket_log.csv", alias="TICKET_LOG_CSV")
    ca_bundle: Path | None = Field(default=None, alias="CA_BUNDLE")

    # Retrieval
    retrieval_mode: RetrievalMode = Field(default=RetrievalMode.HYBRID_GRAPH, alias="RETRIEVAL_MODE")
    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    top_k_per_retriever: int = Field(default=8, alias="TOP_K_PER_RETRIEVER")
    rrf_k: int = Field(default=60, alias="RRF_K")
    fusion_queries: int = Field(default=3, alias="RAG_FUSION_QUERIES")
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    final_context_k: int = Field(default=5, alias="FINAL_CONTEXT_K")
    min_rerank_score: float = Field(default=3.0, alias="MIN_RERANK_SCORE")  # 0-10 scale
    ocr_enabled: bool = Field(default=True, alias="OCR_ENABLED")
    graph_enabled: bool = Field(default=True, alias="GRAPH_ENABLED")

    # Decision thresholds
    confidence_threshold: float = Field(default=0.70, alias="CONFIDENCE_THRESHOLD")
    approval_timeout_s: int = Field(default=600, alias="APPROVAL_TIMEOUT_SECONDS")
    max_agent_messages: int = Field(default=14, alias="MAX_AGENT_MESSAGES")
    max_complaint_chars: int = Field(default=4000, alias="MAX_COMPLAINT_CHARS")

    # Email (Gmail: use an App Password, never the account password)
    support_mailbox: str = Field(default="", alias="SUPPORT_MAILBOX")            # set in .env
    support_mailbox_password: SecretStr = Field(default=SecretStr(""), alias="SUPPORT_MAILBOX_APP_PASSWORD")
    escalation_email: str = Field(default="", alias="ESCALATION_EMAIL")          # set in .env
    smtp_host: str = Field(default="smtp.gmail.com", alias="SMTP_HOST")
    smtp_port: int = Field(default=465, alias="SMTP_PORT")
    imap_host: str = Field(default="imap.gmail.com", alias="IMAP_HOST")
    mail_poll_seconds: int = Field(default=60, alias="MAIL_POLL_SECONDS")
    # Intake filters: only genuine support mail becomes a ticket (see emailer.is_support_mail)
    mail_support_alias: str = Field(default="support", alias="MAIL_SUPPORT_ALIAS")      # <mailbox>+support@gmail.com
    mail_require_alias: bool = Field(default=False, alias="MAIL_REQUIRE_SUPPORT_ALIAS")  # strictest mode
    # True = every personal (non-bulk) mail becomes a ticket. Use for a DEDICATED support inbox.
    mail_accept_all: bool = Field(default=False, alias="MAIL_ACCEPT_ALL")
    mail_accept_keywords: str = Field(
        default=("complaint,support,help,issue,problem,refund,return,replacement,warranty,repair,"
                 "not working,stopped working,doesn't work,won't,broken,defective,damaged,faulty,error,"
                 "not pairing,not charging,not connecting,pair,charge,battery,delivery,delivered,shipment,"
                 "tracking,order,login,password,account,headset,headphone,earphone,earbuds,vacuum,robot,"
                 "mop,fryer,air fryer"),
        alias="MAIL_ACCEPT_KEYWORDS")
    mail_process_existing: bool = Field(default=False, alias="MAIL_PROCESS_EXISTING")  # False = only new mail
    mail_max_per_poll: int = Field(default=10, alias="MAIL_MAX_PER_POLL")
    email_enabled: bool = Field(default=True, alias="EMAIL_ENABLED")
    company_name: str = Field(default="NovaHome Support", alias="COMPANY_NAME")
    timezone: str = Field(default="Asia/Kolkata", alias="TIMEZONE")  # used for every stored timestamp
    # Customer confirmation of a proposed resolution
    confirmation_timeout_hours: int = Field(default=48, alias="CONFIRMATION_TIMEOUT_HOURS")
    confirmation_sweep_minutes: int = Field(default=30, alias="CONFIRMATION_SWEEP_MINUTES")
    support_site_url: str = Field(default="http://localhost:8501", alias="SUPPORT_SITE_URL")

    # Google Sheets (service account JSON shared with the sheet owner's account)
    google_sa_json: Path | None = Field(default=None, alias="GOOGLE_SERVICE_ACCOUNT_JSON")
    google_sheet_id: str = Field(default="", alias="GOOGLE_SHEET_ID")
    google_sheet_tab: str = Field(default="TicketLog", alias="GOOGLE_SHEET_TAB")

    # Admin + API
    admin_username: str = Field(default="admin", alias="ADMIN_USERNAME")
    admin_password: SecretStr = Field(default=SecretStr(""), alias="ADMIN_PASSWORD")  # set in .env; empty = admin login off
    admin_token_secret: SecretStr = Field(default=SecretStr(""), alias="ADMIN_TOKEN_SECRET")
    api_base_url: str = Field(default="http://127.0.0.1:8000", alias="API_BASE_URL")

    # LangSmith (LangChain reads LANGSMITH_* directly from the environment too)
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_project: str = Field(default="buildathon-support-agent", alias="LANGSMITH_PROJECT")

    def resolve(self, p: Path) -> Path:
        """Make a configured path absolute relative to the project root."""
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    def ensure_dirs(self) -> None:
        """Resolve relative paths against the project root and create storage folders."""
        for name in ("data_dir", "index_dir", "cache_dir", "db_path", "log_csv_path"):
            setattr(self, name, self.resolve(getattr(self, name)))
        if self.ca_bundle:
            self.ca_bundle = self.resolve(self.ca_bundle)
        if self.google_sa_json:
            self.google_sa_json = self.resolve(self.google_sa_json)
        for p in (self.index_dir, self.cache_dir, self.db_path.parent, self.log_csv_path.parent):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton so .env is parsed once per process."""
    s = Settings()
    s.ensure_dirs()
    return s


def product_slug(name: str) -> str:
    """Stable folder-safe key for a product name (used for index dirs and graph keys)."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# --------------------------------------------------------------------------- #
# Prompts (kept here so they are versioned in one place)
# --------------------------------------------------------------------------- #
PLANNER_SYSTEM = """You are the query planner of a customer-support RAG system.
Products available (choose exactly one slug, or "unknown"): {products}
Product evidence retrieved from the product catalogue index:
{router_evidence}

Tasks:
1. Rewrite the customer's latest message into ONE standalone English question,
   resolving references using the chat history.
2. Pick the product the customer is talking about ONLY if they name it or clearly describe
   it (e.g. "my headphones", "the robot vacuum", "air fryer"). If they refer to any other
   product, brand or device that is not in the list (e.g. a washing machine, a phone, another
   brand), or give no product at all, use "unknown". The product evidence above is only a
   similarity search - it always returns something, so never pick a product from it alone.
   Put the product name the customer actually used in mentioned_product.
3. Classify category ({categories}), urgency (Low/Medium/High) and sentiment
   (Positive/Neutral/Negative/Very Negative). High urgency = safety risk, money lost,
   legal/fraud threat, or the customer is blocked and very upset.
4. Extract named entities (features, error codes, policy names, order/delivery terms).
5. Choose doc_type filter only when the question clearly targets one document type.
6. Write {n_queries} diverse search queries (RAG-Fusion) covering different phrasings.
7. issue_title: a 3-6 word neutral title a customer would be happy to see in an email
   subject, e.g. "Headphones not pairing", "Refund for damaged item". No emotions,
   no "the customer", no blame.
Never invent facts. Output must follow the schema."""

ANSWER_SYSTEM = """You are a customer-support writer for {company}.
Write ONLY from the numbered CONTEXT passages below. If the context does not
contain the answer, say clearly that you could not find this in our documentation
and that a specialist will follow up. Never invent policies, numbers, dates or links.
Cite sources inline as [S1], [S2] matching the passage numbers.
Reply in language: {language}. Tone: {tone}.

CONTEXT:
{context}"""

JUDGE_SYSTEM = """You are a strict evaluator of a customer-support answer.
Score using ONLY the provided context:
- faithfulness: fraction of factual claims in the answer supported by the context (0-1)
- relevance: how well the answer addresses the question (0-1)
- completeness: whether the important context facts needed were used (0-1)
List any unsupported claims verbatim."""

RERANK_SYSTEM = """Score how useful each passage is for answering the question
(0 = irrelevant, 10 = directly answers it). Judge only on content."""

GRAPH_PLAN_SYSTEM = """You plan read-only traversals over a product knowledge graph.
Schema: (:Product {{slug}})-[:HAS_DOCUMENT]->(:Document {{name, doc_type}})-[:HAS_SECTION]->(:Section {{id, title, text}})
(:Section)-[:MENTIONS]->(:Entity {{name, type, product}})-[:RELATED_TO {{relation}}]->(:Entity)
(:Section)-[:NEXT]->(:Section)
Return the entity names from the question worth looking up (lower-case, as they
would appear in documentation), optional relation keywords and the number of hops (1 or 2)."""

ENTITY_EXTRACT_SYSTEM = """Extract the key entities (features, parts, policies, time
limits, fees, procedures, error codes) and their relations from this support
document section. Use short lower-case names. Only include what the text states."""

QUERY_AGENT_SYS = """You are Customer_Query_Agent. Your ONLY job: call the tool
`analyze_complaint` exactly once with the customer's complaint text, then report
the result as: Product, Category, Urgency, Sentiment, Summary (one line each).
Do not answer the customer. Do not invent anything not in the tool result."""

KNOWLEDGE_AGENT_SYS = """You are Knowledge_Base_Agent. Call `search_knowledge_base`
exactly once using the summary and the product slug given by Customer_Query_Agent.
Then list ONLY the relevant passages, each with its source file name in brackets.
If the tool says nothing relevant was found, say exactly: NO_MATCHING_DOCUMENT.
Never use outside knowledge and never mix products."""

RESOLUTION_AGENT_SYS = """You are Resolution_Agent. Call `draft_customer_reply`
exactly once. Then present the customer-ready reply exactly as returned, followed
by a line 'Confidence: <value>'. Do not add facts. If the tool reports no relevant
information, say so honestly."""

LOG_AGENT_SYS = """You are Log_Agent. First call `record_ticket` exactly once.
- If its result says status=Pending Confirmation (reply sent, waiting for the customer to confirm):
  summarise the log row and end your message with the word LOGGED.
- If its result says status=Escalated and approval is pending: summarise the escalation reasons
  and end with AWAITING_APPROVAL (do NOT write LOGGED yet).
After Support_Approver replies, call `finalize_ticket` with the approver's message, report the
outcome and end with LOGGED."""

SELECTOR_PROMPT = """You coordinate a customer-support team.
Roles:
{roles}

Workflow: Customer_Query_Agent -> Knowledge_Base_Agent -> Resolution_Agent -> Log_Agent.
If Log_Agent says AWAITING_APPROVAL, Support_Approver speaks next, then Log_Agent again.
Read the conversation and select the next role from {participants}. Only return the role.

{history}"""
