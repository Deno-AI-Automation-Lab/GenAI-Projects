"""Pydantic request/response and domain models shared by API, agents and UI."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

Urgency = Literal["Low", "Medium", "High"]
Sentiment = Literal["Positive", "Neutral", "Negative", "Very Negative"]
# Ticket lifecycle:
#   Processing -> Pending Confirmation -> Resolved                (customer confirms)
#                                      -> Escalated - Awaiting Approval (customer disagrees, High priority)
#                                      -> Closed - No Response    (no answer within CONFIRMATION_TIMEOUT_HOURS)
#   Processing -> Escalated - Awaiting Approval -> (human approves) -> Pending Confirmation -> ...
Status = Literal["Processing", "Pending Confirmation", "Resolved", "Escalated", "Escalated - Awaiting Approval",
                 "Closed - No Response", "Closed - Not Supported", "Failed"]
PENDING, RESOLVED, ESCALATED_WAIT, CLOSED_NO_RESPONSE, CLOSED_UNSUPPORTED = (
    "Pending Confirmation", "Resolved", "Escalated - Awaiting Approval", "Closed - No Response",
    "Closed - Not Supported")
Channel = Literal["web", "email", "api"]


_TZ = None


def local_tz():
    """Configured timezone (TIMEZONE in .env, default Asia/Kolkata = IST).
    On Windows zoneinfo needs the `tzdata` package; IST falls back to a fixed +05:30."""
    global _TZ
    if _TZ is None:
        from datetime import timedelta

        from .config import get_settings

        name = get_settings().timezone
        try:
            from zoneinfo import ZoneInfo

            _TZ = ZoneInfo(name)
        except Exception:
            ist = name in ("Asia/Kolkata", "Asia/Calcutta", "IST")
            _TZ = timezone(timedelta(hours=5, minutes=30), "IST") if ist else timezone.utc
    return _TZ


def log_time() -> str:
    """Human-readable timestamp for the ticket log / Google Sheet, e.g. 2026-11-02 19:30 (IST).
    Internal records keep full ISO time (now_iso) for exact auto-close calculations."""
    return datetime.now(local_tz()).strftime("%Y-%m-%d %H:%M")


def now_iso() -> str:
    """Timestamp for all stored records in the configured timezone, e.g. 2026-11-02T15:37:05+05:30."""
    return datetime.now(local_tz()).isoformat(timespec="seconds")


# ----------------------------- API requests ------------------------------- #
class ComplaintRequest(BaseModel):
    """Payload accepted by POST /tickets (UI form, email worker, scripts)."""

    customer_name: str = Field(min_length=1, max_length=120)
    order_id: str | None = Field(default=None, max_length=64,
                                 description="Optional - only when the issue concerns a specific order")
    complaint: str = Field(min_length=5, max_length=4000)
    customer_email: EmailStr | None = None
    channel: Channel = "web"
    session_id: str | None = Field(default=None, description="UI session for chat history")

    @field_validator("customer_name", "complaint")
    @classmethod
    def strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("order_id")
    @classmethod
    def blank_order_is_none(cls, v: str | None) -> str | None:
        """An empty order ID is stored as NULL (blank in the log), never as a placeholder."""
        v = (v or "").strip()
        return v or None


class ChatRequest(BaseModel):
    """Follow-up question about an existing ticket (chatbot)."""

    ticket_id: str
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    """Human decision on an escalated reply."""

    decision: Literal["approve", "edit", "reject"]
    approver: str = "support"
    edited_reply: str | None = None
    note: str | None = None


class ResolveRequest(BaseModel):
    """Support person finishing an escalated ticket that they handled themselves."""

    mode: Literal["confirm", "close"] = Field(
        description="confirm = send the resolution and ask the customer to confirm; close = mark resolved now")
    note: str = Field(default="", max_length=4000, description="What support did - sent to the customer")
    notify_customer: bool = True
    resolver: str = "support"


class AdminLogin(BaseModel):
    username: str
    password: str


# ------------------------------ Planning ---------------------------------- #
class QueryPlan(BaseModel):
    """Structured output of the planning step (RAG-Fusion + filters + triage)."""

    standalone_question: str = Field(description="Self-contained English question")
    product: str = Field(description="Product slug or 'unknown'")
    category: str
    urgency: Urgency
    sentiment: Sentiment
    summary: str = Field(description="One-sentence summary of the problem")
    mentioned_product: str = Field(default="", description="The product/brand/model the customer refers to, "
                                   "copied from their message (e.g. 'Samsung washing machine'); empty if none")
    issue_title: str = Field(default="", description="3-6 word neutral, customer-facing issue title, "
                             "e.g. 'Headphones not pairing'. No emotions, no names, no blame.")
    entities: list[str] = Field(default_factory=list)
    doc_type_filter: str | None = Field(default=None, description="shipping_policy|refund_policy|account_and_login|product_faq or null")
    fusion_queries: list[str] = Field(default_factory=list)


class QueryAnalysis(QueryPlan):
    """QueryPlan + language + deterministic triage signals added in code."""

    language: str = "en"
    language_name: str = "English"
    risk_flags: list[str] = Field(default_factory=list)
    product_name: str = ""


class GraphPlan(BaseModel):
    """LLM-planned graph traversal (validated & executed read-only in code)."""

    entities: list[str] = Field(default_factory=list)
    relation_keywords: list[str] = Field(default_factory=list)
    hops: int = Field(default=1, ge=1, le=2)


# ------------------------------ Retrieval --------------------------------- #
class Chunk(BaseModel):
    """One indexed unit of text with metadata kept as separate fields."""

    chunk_id: str
    product: str
    product_name: str
    doc_type: str
    source_file: str
    section_title: str = ""
    record_id: str = Field(description="Section/record this chunk belongs to (entity)")
    text: str
    prev_id: str | None = None
    next_id: str | None = None
    kind: Literal["text", "table_row", "ocr", "key_facts"] = "text"
    language: str = "en"


class Passage(BaseModel):
    """A retrieved passage returned to agents/UI."""

    chunk_id: str
    source_file: str
    section_title: str
    text: str
    score: float = 0.0
    retrievers: list[str] = Field(default_factory=list)
    rerank_score: float | None = None


class RetrievalResult(BaseModel):
    product: str
    query: str
    passages: list[Passage]
    context: str
    mode: str
    graph_facts: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class RerankScores(BaseModel):
    scores: list[int] = Field(description="One 0-10 score per passage, same order")


class JudgeVerdict(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    unsupported_claims: list[str] = Field(default_factory=list)


class EntityRelation(BaseModel):
    source: str
    relation: str
    target: str


class SectionEntities(BaseModel):
    entities: list[str] = Field(default_factory=list)
    relations: list[EntityRelation] = Field(default_factory=list)


# ----------------------------- Resolution --------------------------------- #
class DraftReply(BaseModel):
    subject: str
    body: str
    confidence: float = Field(ge=0, le=1)
    sources: list[str] = Field(default_factory=list)
    found_relevant_info: bool = True
    guardrail_issues: list[str] = Field(default_factory=list)


class EscalationDecision(BaseModel):
    escalate: bool
    reasons: list[str] = Field(default_factory=list)


class TicketLogRow(BaseModel):
    """Exact columns written to Google Sheets / CSV."""

    ticket_id: str
    timestamp: str
    customer_name: str
    order_id: str
    complaint: str
    category: str
    urgency: str
    sentiment: str
    kb_sources: str
    response: str
    status: str
    escalation_reason: str

    @classmethod
    def columns(cls) -> list[str]:
        return list(cls.model_fields.keys())


class AgentEvent(BaseModel):
    """One step shown in the live timeline."""

    ticket_id: str
    seq: int
    agent: str
    kind: Literal["message", "tool_call", "tool_result", "system", "error"]
    content: str
    created_at: str = Field(default_factory=now_iso)


class TicketResponse(BaseModel):
    ticket_id: str
    status: Status
    customer_name: str
    order_id: str | None = None
    customer_email: str | None = None
    complaint: str
    channel: str = "web"
    product: str | None = None
    category: str | None = None
    urgency: str | None = None
    sentiment: str | None = None
    language: str | None = None
    summary: str | None = None
    reply_subject: str | None = None
    reply_body: str | None = None
    confidence: float | None = None
    kb_sources: list[str] = Field(default_factory=list)
    escalation_reasons: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    confirmation_requested_at: str | None = None
    customer_feedback: str | None = None
    access_code: str | None = Field(default=None, description="With the ticket ID, opens the ticket in 'My tickets'")


class ConfirmRequest(BaseModel):
    """Customer's answer to 'Did this solve your issue?' (UI buttons)."""

    resolved: bool
    comment: str | None = Field(default=None, max_length=2000)


class CustomerReplyIntent(BaseModel):
    """LLM fallback classification of a customer's email reply to a proposed resolution."""

    intent: Literal["confirmed", "not_resolved", "question"]
    reason: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    standalone_question: str
