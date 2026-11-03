"""End-to-end ticket processing (used by FastAPI, the mail worker, scripts and tests).

process_ticket():
  1. create ticket in long-term memory (ticket id returned immediately to the caller)
  2. input guardrails (PII masking, prompt-injection neutralising)
  3. AutoGen team run (live events streamed to memory)
  4. fallback: any step the team skipped/failed is executed deterministically, so a
     ticket ALWAYS ends Resolved or Escalated and is always logged
"""
from __future__ import annotations

import asyncio
import logging

from langsmith import traceable

from .agents import (APPROVER, LOGGER, TicketContext, _record_ticket_sync, apply_decision, make_tools,
                     run_agent_team)
from .guardrails import check_input
from .metrics import format_metrics, track_usage
from .models import ChatResponse, ComplaintRequest, DraftReply, QueryAnalysis
from .planner import plan_query
from .retrieval import generate_answer, hybrid_retrieve, sources_of
from .services import SupportServices

log = logging.getLogger(__name__)


def create_ticket(svc: SupportServices, req: ComplaintRequest) -> TicketContext:
    """Persist the ticket and return its context (complaint already guard-railed)."""
    guard = check_input(req.complaint, svc.settings.max_complaint_chars)
    tid = svc.memory.create_ticket(customer_name=req.customer_name, order_id=req.order_id,
                                   complaint=guard.text, customer_email=req.customer_email, channel=req.channel)
    svc.memory.add_chat(tid, "user", guard.text)
    ctx = TicketContext(services=svc, ticket_id=tid, customer_name=req.customer_name, order_id=req.order_id,
                        complaint=guard.text, customer_email=req.customer_email, channel=req.channel)
    ctx.emit("Input_Guardrail", "system",
             "Input accepted" + (f" (issues: {', '.join(guard.issues)})" if guard.issues else "")
             if guard.allowed else f"Input rejected: {', '.join(guard.issues)}")
    if not guard.allowed:
        svc.memory.update_ticket(tid, status="Failed", reply_body="Please describe your issue in a few words.")
    return ctx


async def _run_fallback(ctx: TicketContext) -> None:
    """Deterministically complete any missing workflow step (same tools as the agents)."""
    tools = make_tools(ctx)
    if ctx.analysis is None:
        ctx.emit("Fallback", "system", "Running Customer Query step directly")
        await tools["analyze_complaint"](ctx.complaint)
    if ctx.retrieval is None:
        ctx.emit("Fallback", "system", "Running Knowledge Base step directly")
        await tools["search_knowledge_base"](ctx.analysis.summary, ctx.analysis.product)
    if ctx.draft is None:
        ctx.emit("Fallback", "system", "Running Resolution step directly")
        await tools["draft_customer_reply"](ctx.analysis.summary)
    if not ctx.recorded:
        ctx.emit("Fallback", "system", "Running Log step directly")
        result = await asyncio.to_thread(_record_ticket_sync, ctx)
        ctx.emit(LOGGER, "tool_result", result)


async def _wait_for_approval(ctx: TicketContext) -> None:
    """If the team stopped before the approver answered, wait here (same timeout)."""
    from .agents import make_approval_input, parse_approver_message

    msg = await make_approval_input(ctx)("approval")
    decision, payload = parse_approver_message(msg)
    if decision == "edit" and not payload:
        payload = (ctx.services.memory.get_approval(ctx.ticket_id) or {}).get("edited_reply")
    ctx.emit(APPROVER, "message", msg)
    ctx.emit(LOGGER, "tool_result", await asyncio.to_thread(apply_decision, ctx, decision, payload))


@traceable(name="process_ticket", run_type="chain")
async def process_ticket(svc: SupportServices, ctx: TicketContext) -> str:
    """Run the full workflow for an existing ticket context. Never raises."""
    if svc.memory.get_ticket(ctx.ticket_id).status == "Failed":
        return "Failed"
    with track_usage() as usage:   # latency, tokens and approximate cost of the automated part
        try:
            if svc.settings.openai_api_key.get_secret_value():
                await run_agent_team(ctx)
            else:
                ctx.emit("System", "error", "OPENAI_API_KEY missing – agent team skipped")
        except Exception as exc:
            log.exception("Agent team failed for %s", ctx.ticket_id)
            ctx.emit("System", "error", f"Agent team error: {exc.__class__.__name__}: {exc}. Using fallback.")
        try:
            if not ctx.recorded:
                await _run_fallback(ctx)
        except Exception as exc:
            log.exception("Fallback failed for %s", ctx.ticket_id)
            ctx.emit("System", "error", f"Fallback error: {exc}")
            if not ctx.recorded:
                svc.memory.update_ticket(ctx.ticket_id, status="Escalated - Awaiting Approval",
                                         escalation_reasons=[f"System error: {exc}"])
                svc.memory.request_approval(ctx.ticket_id)
    _save_metrics(ctx, usage.summary())   # measured before any wait for a human decision
    try:
        if ctx.awaiting_approval and not ctx.finalized:
            await _wait_for_approval(ctx)
    except Exception as exc:
        log.exception("Waiting for approval failed for %s", ctx.ticket_id)
        ctx.emit("System", "error", f"Approval step error: {exc}")
        # Last resort: make sure humans see it
        if not ctx.recorded:
            svc.memory.update_ticket(ctx.ticket_id, status="Escalated - Awaiting Approval",
                                     escalation_reasons=[f"System error: {exc}"])
            svc.memory.request_approval(ctx.ticket_id)
    status = svc.memory.get_ticket(ctx.ticket_id).status
    ctx.emit("System", "system", f"Ticket finished with status: {status}")
    return status


def _save_metrics(ctx: TicketContext, metrics: dict) -> None:
    """Store processing time / tokens / cost with the ticket (admin dashboard) and show the
    time in the live timeline. Never breaks the ticket if storing fails."""
    try:
        extra = ctx.services.memory.get_extra(ctx.ticket_id) or {}
        extra["metrics"] = metrics
        ctx.services.memory.update_ticket(ctx.ticket_id, extra=extra)
        ctx.emit("System", "system", f"Processing time {metrics['latency_s']:.1f} s · "
                                     f"{metrics['llm_calls']} model calls")
        log.info("Ticket %s metrics: %s", ctx.ticket_id, format_metrics(metrics))
    except Exception as exc:  # metrics are best-effort
        log.warning("Could not store metrics for %s: %s", ctx.ticket_id, exc)


def context_from_memory(svc: SupportServices, ticket_id: str) -> TicketContext:
    """Rebuild a TicketContext from stored state (used by the approval API)."""
    t = svc.memory.get_ticket(ticket_id)
    if t is None:
        raise KeyError(ticket_id)
    extra = svc.memory.get_extra(ticket_id)
    ctx = TicketContext(services=svc, ticket_id=t.ticket_id, customer_name=t.customer_name, order_id=t.order_id,
                        complaint=t.complaint, customer_email=t.customer_email, channel=t.channel)
    if extra.get("analysis"):
        ctx.analysis = QueryAnalysis(**extra["analysis"])
    if extra.get("draft"):
        ctx.draft = DraftReply(**extra["draft"])
    if extra.get("decision"):
        from .models import EscalationDecision

        ctx.decision = EscalationDecision(**extra["decision"])
    ctx.recorded, ctx.awaiting_approval = True, t.status == "Escalated - Awaiting Approval"
    return ctx


def finalize_after_pipeline(svc: SupportServices, ticket_id: str) -> str | None:
    """Called by the approval API: if the pipeline is no longer waiting (timeout/restart),
    apply the stored decision directly."""
    appr = svc.memory.get_approval(ticket_id)
    if not appr or appr.get("state") != "decided":
        return None
    ctx = context_from_memory(svc, ticket_id)
    if not ctx.awaiting_approval:
        return None
    payload = appr.get("edited_reply") if appr["decision"] == "edit" else appr.get("note")
    result = apply_decision(ctx, appr["decision"], payload)
    if result.startswith("decision already"):
        return None  # the live agent run finalized it
    ctx.emit(LOGGER, "tool_result", f"(finalized from admin console) {result}")
    return result


@traceable(name="answer_followup", run_type="chain")
def answer_followup(svc: SupportServices, ticket_id: str, message: str) -> ChatResponse:
    """Chatbot follow-up on a ticket: plan with chat history -> retrieve -> grounded answer."""
    t = svc.memory.get_ticket(ticket_id)
    if t is None:
        raise KeyError(ticket_id)
    guard = check_input(message, 2000)
    if not guard.allowed:
        return ChatResponse(answer="Could you add a little more detail?", sources=[], standalone_question=message)
    history = svc.memory.chat_history(ticket_id)
    svc.memory.add_chat(ticket_id, "user", guard.text)
    router = svc.router
    plan = plan_query(guard.text, history, router.products, [], svc.settings)
    if plan.product == "unknown" and t.product:
        plan.product = t.product  # stay on the ticket's product
    if plan.product in ("unknown", None):
        answer = "I couldn't tell which product this is about. Could you mention the product name?"
        svc.memory.add_chat(ticket_id, "assistant", answer)
        return ChatResponse(answer=answer, sources=[], standalone_question=plan.standalone_question)
    res = hybrid_retrieve(svc.registry.get(plan.product), plan, svc.settings)
    from .language import detect_language, language_name

    lang = language_name(detect_language(guard.text))
    answer = generate_answer(plan.standalone_question, res, lang, plan.urgency, svc.settings.company_name) or (
        "I could not find this in our documentation. Our support team will help you on this ticket.")
    from .guardrails import check_output

    out = check_output(answer, sources_of(res), res.context)
    svc.memory.add_chat(ticket_id, "assistant", out.text)
    return ChatResponse(answer=out.text, sources=sources_of(res), standalone_question=plan.standalone_question)
