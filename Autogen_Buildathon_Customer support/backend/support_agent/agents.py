"""AutoGen multi-agent team: Query → Knowledge → Resolution → Log (+ human approver).

Design for production reliability:
* Each AssistantAgent owns ONE typed tool that does the real, deterministic, grounded
  work and writes a structured result into `TicketContext`. The agent's own text
  (reflect_on_tool_use=True) is a readable summary for the live timeline. Business
  data is never parsed back out of free text.
* SelectorGroupChat uses an LLM selector; `candidate_func` narrows the candidates to
  the agents that are valid for the current workflow state, which enforces the
  Query → Knowledge → Resolution → Log order (and Approver → Log after escalation).
* Termination: TextMentionTermination("LOGGED", sources=["Log_Agent"]) OR a
  MaxMessageTermination safety net.
* Human-in-the-loop: AutoGen UserProxyAgent `Support_Approver` whose input_func waits
  for a decision submitted from the admin console (via the API / MemoryStore).
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core import CancellationToken
from langsmith import traceable

from .config import (KNOWLEDGE_AGENT_SYS, LOG_AGENT_SYS, QUERY_AGENT_SYS, RESOLUTION_AGENT_SYS,
                     SELECTOR_PROMPT, URGENCY_SCORE)
from .emailer import build_subject, render_customer_email, render_escalation_email, send_email
from .models import CLOSED_UNSUPPORTED, PENDING, DraftReply, EscalationDecision, QueryAnalysis, RetrievalResult, TicketLogRow, log_time, now_iso
from .planner import analyze_complaint
from .resolution import draft_reply, evaluate_escalation, policy_escalation_evidence
from .retrieval import hybrid_retrieve
from .services import SupportServices

log = logging.getLogger(__name__)

QUERY, KNOWLEDGE, RESOLUTION, LOGGER, APPROVER = (
    "Customer_Query_Agent", "Knowledge_Base_Agent", "Resolution_Agent", "Log_Agent", "Support_Approver")


@dataclass
class TicketContext:
    """Per-ticket workflow state shared by the tools (one instance per run)."""

    services: SupportServices
    ticket_id: str
    customer_name: str
    order_id: str | None
    complaint: str
    customer_email: Optional[str]
    channel: str = "web"
    analysis: Optional[QueryAnalysis] = None
    retrieval: Optional[RetrievalResult] = None
    draft: Optional[DraftReply] = None
    decision: Optional[EscalationDecision] = None
    recorded: bool = False
    awaiting_approval: bool = False
    finalized: bool = False
    final_status: str = "Processing"
    step_errors: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def emit(self, agent: str, kind: str, content: str) -> None:
        self.services.memory.add_event(self.ticket_id, agent, kind, content)

    def persist(self) -> None:
        """Save structured state to long-term memory (also used by the approval API)."""
        a, d, r = self.analysis, self.draft, self.retrieval
        # Merge into what is stored: a context rebuilt from memory (approval API) has no
        # retrieval object and must not wipe the passages saved during the agent run.
        extra = self.services.memory.get_extra(self.ticket_id) or {}
        if a:
            extra["analysis"] = a.model_dump()
        if d:
            extra["draft"] = d.model_dump()
        if self.decision:
            extra["decision"] = self.decision.model_dump()
        if r:
            extra.update(retrieval_mode=r.mode, retrieval_errors=r.errors,
                         passages=[p.model_dump() for p in r.passages], graph_facts=r.graph_facts)
        fields: dict[str, Any] = {"extra": extra}
        if a:
            fields.update(product=a.product, category=a.category, urgency=a.urgency, sentiment=a.sentiment,
                          language=a.language, summary=a.summary)
        if d:
            fields.update(confidence=d.confidence, kb_sources=d.sources)
            current = self.services.memory.get_ticket(self.ticket_id)
            if current is not None and not current.reply_subject:   # draft only until a reply is sent
                fields["reply_body"] = d.body
        if self.decision:
            fields["escalation_reasons"] = self.decision.reasons
        self.services.memory.update_ticket(self.ticket_id, **fields)


# --------------------------------------------------------------------------- #
# Tool factory (closures keep per-ticket state; signatures/docstrings are what
# AutoGen exposes to the model as the tool schema)
# --------------------------------------------------------------------------- #
def make_tools(ctx: TicketContext) -> dict[str, Callable[..., Awaitable[str]]]:
    svc, s = ctx.services, ctx.services.settings

    async def analyze_complaint_tool(complaint_text: str) -> str:
        """Classify a customer complaint.

        Uses RAG over the product catalogue to identify which product the customer
        means, then returns JSON with product, category, urgency (Low/Medium/High),
        sentiment, a one-line summary, detected language and risk flags.

        Args:
            complaint_text: The customer's raw complaint text.
        """
        history = [{"role": "user", "content": f"Earlier ticket {h['ticket_id']} ({h['status']}): {h['summary']}"}
                   for h in svc.memory.customer_history(email=ctx.customer_email, order_id=ctx.order_id,
                                                        exclude=ctx.ticket_id) if h.get("summary")]
        # Always analyse the guarded complaint held in context (the model may paraphrase)
        text = ctx.complaint if len(complaint_text) < len(ctx.complaint) // 2 else complaint_text
        ctx.analysis = await asyncio.to_thread(analyze_complaint, text, svc.router, s, history)
        ctx.persist()
        a = ctx.analysis
        return json.dumps({"product": a.product, "product_name": a.product_name, "category": a.category,
                           "urgency": a.urgency, "sentiment": a.sentiment, "summary": a.summary,
                           "language": a.language_name, "risk_flags": a.risk_flags}, ensure_ascii=False)

    async def search_knowledge_base(query: str, product: str) -> str:
        """Search ONLY the given product's documents (shipping_policy, refund_policy,
        account_and_login, product_faq) with hybrid RAG: RAG-Fusion queries, vector +
        BM25 + entity-summary + exact-entity + knowledge-graph retrieval, Reciprocal Rank
        Fusion and re-ranking. Returns the relevant passages, each with its source file
        name, or NO_MATCHING_DOCUMENT if nothing relevant exists.

        Args:
            query: Short description of the customer's problem (summary + category).
            product: Product slug chosen by the Customer Query Agent.
        """
        if ctx.analysis is None:
            return "ERROR: analyze the complaint first."
        a = ctx.analysis
        # Product isolation: never search another product's documents.
        if product != a.product:
            ctx.emit(KNOWLEDGE, "system", f"Product '{product}' overridden to routed product '{a.product}'.")
        if a.product == "unknown":
            ctx.retrieval = RetrievalResult(product="unknown", query=query, passages=[], context="", mode="none",
                                            errors=["product could not be identified"])
            ctx.persist()
            return "NO_MATCHING_DOCUMENT: product could not be identified from the complaint."
        plan = a.model_copy(update={"fusion_queries": list(dict.fromkeys(a.fusion_queries + [query]))})
        idx = await asyncio.to_thread(svc.registry.get, a.product)
        ctx.retrieval = await asyncio.to_thread(hybrid_retrieve, idx, plan, s)
        ctx.persist()
        if not ctx.retrieval.passages:
            return "NO_MATCHING_DOCUMENT: no relevant passage found in this product's documents."
        lines = [f"[{p.source_file} › {p.section_title}] {p.text[:600]}" for p in ctx.retrieval.passages]
        if ctx.retrieval.graph_facts:
            lines.append("Graph facts: " + "; ".join(ctx.retrieval.graph_facts[:6]))
        return "\n\n".join(lines)

    async def draft_customer_reply(summary: str) -> str:
        """Write the customer-ready reply using ONLY the passages found by the Knowledge
        Base Agent, in the customer's language and a tone matched to ticket priority.
        Returns the reply text and a confidence score between 0 and 1.

        Args:
            summary: One-line summary of the customer's problem.
        """
        if ctx.analysis is None:
            return "ERROR: analyze the complaint first."
        if ctx.retrieval is None:
            await search_knowledge_base(summary, ctx.analysis.product)
        supported = [p["name"] for p in svc.router.products]
        ctx.draft = await asyncio.to_thread(draft_reply, ctx.analysis, ctx.retrieval, s, None, None, ctx.order_id,
                                            supported)
        ctx.persist()
        d = ctx.draft
        status = "NO RELEVANT INFORMATION FOUND" if not d.found_relevant_info else "draft ready"
        return f"{status}\nConfidence: {d.confidence:.2f}\n\n{d.body}"

    async def record_ticket(note: str = "") -> str:
        """Apply the escalation rules (High urgency, very angry customer, no matching
        document, legal/fraud/payment dispute, low confidence) with a RAG check of the
        product's policies, write the ticket row to the Google Sheet log, and send the
        resolution email or the escalation email. Returns status and reasons.

        Args:
            note: Optional remark from the Log Agent.
        """
        if ctx.analysis is None:
            return "ERROR: analyze the complaint first."
        return await asyncio.to_thread(_record_ticket_sync, ctx)

    async def finalize_ticket(approver_message: str) -> str:
        """Apply the human approver's decision on an escalated ticket (APPROVE, EDIT: <new
        reply>, REJECT: <reason>, or TIMEOUT), update the log and email the customer.

        Args:
            approver_message: Exactly what Support_Approver replied.
        """
        return await asyncio.to_thread(_finalize_from_message, ctx, approver_message)

    analyze_complaint_tool.__name__ = "analyze_complaint"  # tool name shown to the model
    return {"analyze_complaint": analyze_complaint_tool, "search_knowledge_base": search_knowledge_base,
            "draft_customer_reply": draft_customer_reply, "record_ticket": record_ticket,
            "finalize_ticket": finalize_ticket}


# --------------------------------------------------------------------------- #
# Log-agent business logic (sync, reused by the approval API and fallbacks)
# --------------------------------------------------------------------------- #
def build_log_row(ctx: TicketContext, status: str, response: str) -> TicketLogRow:
    a, d = ctx.analysis, ctx.draft
    return TicketLogRow(
        ticket_id=ctx.ticket_id, timestamp=log_time(), customer_name=ctx.customer_name, order_id=ctx.order_id or "",
        complaint=ctx.complaint, category=a.category if a else "", urgency=a.urgency if a else "",
        sentiment=a.sentiment if a else "", kb_sources="; ".join(d.sources if d else []),
        response=response, status=status,
        escalation_reason="; ".join(ctx.decision.reasons) if ctx.decision and ctx.decision.escalate else "",
    )


def _send_customer_reply(ctx: TicketContext, tag: str, body: str, escalated: bool) -> str:
    """Email the proposed resolution (standard template, neutral subject, YES/NO instructions)."""
    from .lifecycle import deliver_reply

    ctx.persist()  # make sure analysis (product, issue title, language) is stored first
    note = deliver_reply(ctx.services, ctx.ticket_id, tag=tag, body=body,
                         sources=ctx.draft.sources if ctx.draft else [], escalated=escalated)
    log.info("%s: reply delivery -> %s", ctx.ticket_id, note)
    ctx.emit(LOGGER, "error" if "failed" in note else "system", f"Customer reply: {note}")
    return note


def _record_ticket_sync(ctx: TicketContext) -> str:
    svc, s = ctx.services, ctx.services.settings
    idx = None
    if ctx.analysis and ctx.analysis.product != "unknown":
        try:
            idx = svc.registry.get(ctx.analysis.product)
        except Exception as exc:
            ctx.emit(LOGGER, "error", f"Index unavailable for policy check: {exc}")
    evidence = policy_escalation_evidence(idx, ctx.complaint)
    ctx.decision = evaluate_escalation(ctx.analysis, ctx.retrieval, ctx.draft, s, evidence)
    body = ctx.draft.body if ctx.draft else ""
    if not ctx.decision.escalate:
        # Not closed yet: the customer must confirm the resolution (email YES/NO or UI buttons)
        from .emailer import TAG_PROPOSED
        from .lifecycle import request_confirmation

        delivery = _send_customer_reply(ctx, TAG_PROPOSED, body, escalated=False)
        request_confirmation(svc, ctx.ticket_id, body)
        ctx.final_status, ctx.finalized, ctx.recorded = PENDING, True, True
        return f"status=Pending Confirmation; logged; {delivery}"

    # Escalated: notify humans, hold the customer reply for approval
    subj, mail = render_escalation_email(
        ticket_id=ctx.ticket_id, customer_name=ctx.customer_name, customer_email=ctx.customer_email,
        order_id=ctx.order_id, complaint=ctx.complaint, analysis=ctx.analysis.model_dump(),
        reasons=ctx.decision.reasons, draft_reply=body, sources=ctx.draft.sources if ctx.draft else [])
    try:
        send_email(s, s.escalation_email, subj, mail)
        esc_note = f"escalation email sent to {s.escalation_email}"
    except Exception as exc:
        ctx.emit(LOGGER, "error", f"Escalation email failed: {exc}")
        esc_note = f"escalation email failed ({exc.__class__.__name__}); visible in admin console"
    ctx.final_status = "Escalated - Awaiting Approval"
    where = svc.logger.upsert(build_log_row(ctx, ctx.final_status, body))
    ctx.persist()   # reasons, analysis and draft first: nobody may see "Escalated" without its reasons
    svc.memory.request_approval(ctx.ticket_id)
    svc.memory.update_ticket(ctx.ticket_id, status=ctx.final_status)
    ctx.recorded, ctx.awaiting_approval = True, True
    return (f"status=Escalated; approval pending; logged to {where}; {esc_note}; "
            f"reasons: {' | '.join(ctx.decision.reasons)}")


def parse_approver_message(msg: str) -> tuple[str, str | None]:
    """'APPROVE' | 'EDIT: text' | 'REJECT: reason' | 'TIMEOUT' -> (decision, payload)."""
    m = (msg or "").strip()
    head, _, rest = m.partition(":")
    head = head.strip().upper()
    if head.startswith("APPROVE"):
        return "approve", None
    if head.startswith("EDIT"):
        return "edit", rest.strip() or None
    if head.startswith("REJECT"):
        return "reject", rest.strip() or None
    return "timeout", None


def apply_decision(ctx: TicketContext, decision: str, payload: str | None) -> str:
    """Finalize an escalated ticket. Shared by the agent tool and the admin API."""
    svc = ctx.services
    if decision not in ("approve", "edit", "reject"):
        # timeout: stays pending; the admin console can still approve later
        return "approval timed out; ticket remains 'Escalated - Awaiting Approval'"
    appr = svc.memory.get_approval(ctx.ticket_id) or {}
    if appr.get("state") == "pending":
        # The model must not approve on its own - only a human decision counts
        return "no human decision yet; ticket stays 'Escalated - Awaiting Approval'"
    if not svc.memory.claim_approval(ctx.ticket_id):
        ctx.finalized = True
        return "decision already applied (finalized from the admin console)"
    ctx.finalized = True
    if decision in ("approve", "edit"):
        from .emailer import TAG_ESCALATED
        from .lifecycle import request_confirmation

        body = payload if decision == "edit" and payload else (ctx.draft.body if ctx.draft else "")
        if ctx.analysis is not None and ctx.analysis.product == "unknown":
            # Unsupported / unidentified product: nothing to confirm - send and close
            from .lifecycle import close_unsupported

            delivery = close_unsupported(svc, ctx.ticket_id, body)
            ctx.final_status = CLOSED_UNSUPPORTED
            return f"decision={decision}; status={CLOSED_UNSUPPORTED}; {delivery}"
        delivery = _send_customer_reply(ctx, TAG_ESCALATED, body, escalated=True)
        request_confirmation(svc, ctx.ticket_id, body)   # approved reply also waits for the customer
        ctx.final_status = PENDING
        return f"decision={decision}; status={PENDING}; {delivery}"
    status = "Escalated"
    body = f"[Reply rejected by support - handled manually] {payload or ''}".strip()
    delivery = "customer reply withheld; support will contact the customer directly"
    ctx.final_status = status
    svc.logger.upsert(build_log_row(ctx, status, body))
    svc.memory.update_ticket(ctx.ticket_id, status=status)
    return f"decision={decision}; status={status}; {delivery}"


def _finalize_from_message(ctx: TicketContext, approver_message: str) -> str:
    decision, payload = parse_approver_message(approver_message)
    if decision in ("approve", "edit", "reject"):
        appr = ctx.services.memory.get_approval(ctx.ticket_id) or {}
        if decision == "edit" and not payload:
            payload = appr.get("edited_reply")
    return apply_decision(ctx, decision, payload)


# --------------------------------------------------------------------------- #
# Human approver input (UserProxyAgent)
# --------------------------------------------------------------------------- #
def make_approval_input(ctx: TicketContext, poll_s: float = 2.0):
    """input_func for UserProxyAgent: waits for a decision posted via the admin API."""

    async def _input(prompt: str, cancellation_token: Optional[CancellationToken] = None) -> str:
        deadline = asyncio.get_event_loop().time() + ctx.services.settings.approval_timeout_s
        ctx.emit(APPROVER, "system", "Waiting for a support person to approve, edit or reject the reply "
                                     "(Admin → Approvals).")
        while asyncio.get_event_loop().time() < deadline:
            if cancellation_token is not None and cancellation_token.is_cancelled():
                break
            appr = ctx.services.memory.get_approval(ctx.ticket_id)
            if appr and appr.get("state") in ("decided", "consumed") and appr.get("decision"):
                d = appr["decision"]
                who = appr.get("approver") or "support"
                if d == "edit":
                    return f"EDIT: {appr.get('edited_reply') or ''}  (by {who})"
                if d == "reject":
                    return f"REJECT: {appr.get('note') or 'no reason given'}  (by {who})"
                return f"APPROVE (by {who})"
            await asyncio.sleep(poll_s)
        return "TIMEOUT"

    return _input


# --------------------------------------------------------------------------- #
# Team construction
# --------------------------------------------------------------------------- #
def next_candidates(ctx: TicketContext, last_speaker: str | None) -> list[str]:
    """Valid next speakers for the current workflow state (Query→Knowledge→Resolution→Log)."""
    if ctx.analysis is None:
        return [QUERY]
    if ctx.retrieval is None:
        return [KNOWLEDGE]
    if ctx.draft is None:
        return [RESOLUTION]
    if not ctx.recorded:
        return [LOGGER]
    if ctx.awaiting_approval and not ctx.finalized:
        return [APPROVER] if last_speaker == LOGGER else [LOGGER]
    return [LOGGER]


def build_model_client(settings):
    """AutoGen's OpenAI chat client (AutoGen agents need an AutoGen ChatCompletionClient;
    all RAG/planning/judging LLM calls use LangChain's ChatOpenAI)."""
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    return OpenAIChatCompletionClient(model=settings.agent_model,
                                      api_key=settings.openai_api_key.get_secret_value(),
                                      temperature=0, timeout=settings.llm_timeout_s,
                                      max_retries=settings.llm_max_retries)


def build_team(ctx: TicketContext, model_client) -> SelectorGroupChat:
    tools = make_tools(ctx)
    query_agent = AssistantAgent(
        QUERY, model_client=model_client, tools=[tools["analyze_complaint"]], reflect_on_tool_use=True,
        description="Understands the complaint: product, category, urgency, sentiment and summary.",
        system_message=QUERY_AGENT_SYS)
    knowledge_agent = AssistantAgent(
        KNOWLEDGE, model_client=model_client, tools=[tools["search_knowledge_base"]], reflect_on_tool_use=True,
        description="Searches only the identified product's FAQs/policies and returns passages with file names.",
        system_message=KNOWLEDGE_AGENT_SYS)
    resolution_agent = AssistantAgent(
        RESOLUTION, model_client=model_client, tools=[tools["draft_customer_reply"]], reflect_on_tool_use=True,
        description="Writes a polite, grounded customer reply with a confidence score.",
        system_message=RESOLUTION_AGENT_SYS)
    log_agent = AssistantAgent(
        LOGGER, model_client=model_client, tools=[tools["record_ticket"], tools["finalize_ticket"]],
        reflect_on_tool_use=True, max_tool_iterations=2,
        description="Logs every ticket, decides escalation and finalizes after human approval.",
        system_message=LOG_AGENT_SYS)
    approver = UserProxyAgent(APPROVER, description="Human support person who approves escalated replies.",
                              input_func=make_approval_input(ctx))

    def candidate_func(messages) -> list[str]:
        last = next((m.source for m in reversed(messages) if getattr(m, "source", None) != "user"), None)
        return next_candidates(ctx, last)

    termination = (TextMentionTermination("LOGGED", sources=[LOGGER])
                   | MaxMessageTermination(ctx.services.settings.max_agent_messages))
    return SelectorGroupChat(
        [query_agent, knowledge_agent, resolution_agent, log_agent, approver],
        model_client=model_client, termination_condition=termination, selector_prompt=SELECTOR_PROMPT,
        candidate_func=candidate_func, allow_repeated_speaker=True)


def task_message(ctx: TicketContext) -> str:
    return (f"New support ticket {ctx.ticket_id} via {ctx.channel}.\n"
            f"Customer name: {ctx.customer_name}\nOrder ID: {ctx.order_id or 'not provided'}\n"
            f"Complaint:\n{ctx.complaint}")


def _event_text(msg: Any) -> tuple[str, str]:
    """Map AutoGen stream items to (kind, text) for the timeline."""
    name = type(msg).__name__
    kind = {"ToolCallRequestEvent": "tool_call", "ToolCallExecutionEvent": "tool_result"}.get(name, "message")
    try:
        text = msg.to_text()
    except Exception:
        text = str(getattr(msg, "content", msg))
    return kind, text


@traceable(name="run_agent_team", run_type="chain")
async def run_agent_team(ctx: TicketContext) -> None:
    """Run the SelectorGroupChat, streaming every step into memory; always closes the client."""
    model_client = build_model_client(ctx.services.settings)
    model_calls = 0
    try:
        team = build_team(ctx, model_client)
        async for msg in team.run_stream(task=task_message(ctx)):
            if type(msg).__name__ in ("TextMessage", "ToolCallRequestEvent") and \
                    str(getattr(msg, "source", "")).endswith("Agent"):
                model_calls += 1   # each of these is one AutoGen model response
            if type(msg).__name__ == "TaskResult":
                ctx.emit("System", "system", f"Conversation stopped: {getattr(msg, 'stop_reason', '')}")
                continue
            if type(msg).__name__ in ("SelectSpeakerEvent", "SelectorEvent"):
                continue
            source = getattr(msg, "source", "System")
            if source == "user":
                continue
            kind, text = _event_text(msg)
            ctx.emit(source, kind, text[:6000])
    finally:
        _record_agent_usage(ctx, model_client, model_calls)
        await model_client.close()  # always release HTTP connections


def _record_agent_usage(ctx: TicketContext, model_client, calls: int) -> None:
    """Add the AutoGen client's token usage to the ticket's cost tracker (best-effort)."""
    from .metrics import current_tracker

    tracker = current_tracker()
    try:
        usage = model_client.total_usage()
        if tracker is not None and usage is not None:
            tracker.add(ctx.services.settings.agent_model, usage.prompt_tokens, usage.completion_tokens, calls)
    except Exception as exc:  # replay/test clients may not report usage
        log.debug("Agent usage not available: %s", exc)


def urgency_score(urgency: str | None) -> int:
    return URGENCY_SCORE.get(urgency or "", 0)
