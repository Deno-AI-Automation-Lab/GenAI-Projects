"""FastAPI backend.  Run:  uvicorn support_agent.api:app --port 8000

Public:  POST /tickets; GET ticket / events / chat with ticket ID + access code (or order ID)
Admin :  POST /admin/login -> bearer token; ticket log (+CSV), dashboard, approvals,
         chunk & graph viewers. Every /admin/* route requires the token.
"""
from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import hmac
import io
import json
import logging
import secrets
import time
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import URGENCY_SCORE, get_settings
from .llm import bootstrap_runtime
from .models import (AdminLogin, AgentEvent, ApprovalRequest, ChatRequest, ChatResponse, ComplaintRequest, ConfirmRequest, ResolveRequest,
                     TicketLogRow, TicketResponse)
from .pipeline import answer_followup, create_ticket, finalize_after_pipeline, process_ticket
from .memory import MemoryStoreError
from .services import SupportServices, build_services

log = logging.getLogger(__name__)
_tasks: set[asyncio.Task] = set()   # strong refs so background runs are not GC'd
_svc: SupportServices | None = None
_TOKEN_TTL = 8 * 3600


def services() -> SupportServices:
    if _svc is None:
        raise HTTPException(503, "Service not initialised")
    return _svc


def spawn(coro) -> None:
    t = asyncio.create_task(coro)
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


async def submit_ticket(svc: SupportServices, req: ComplaintRequest) -> str:
    """Create ticket synchronously, process it in the background; returns ticket id."""
    ctx = create_ticket(svc, req)
    spawn(process_ticket(svc, ctx))
    return ctx.ticket_id


def send_ack(svc: SupportServices, tid: str, req: ComplaintRequest, in_reply_to: str) -> None:
    """Email the ticket ID and access code back to an email customer."""
    from .emailer import render_ack_email, send_email

    t = svc.memory.get_ticket(tid)
    subject, body = render_ack_email(ticket_id=tid, customer_name=req.customer_name, order_id=req.order_id,
                                     access_code=t.access_code, site_url=svc.settings.support_site_url,
                                     company=svc.settings.company_name)
    try:
        send_email(svc.settings, req.customer_email, subject, body, in_reply_to=in_reply_to)
        svc.memory.add_event(tid, "Email_Intake", "system", f"Acknowledgement sent to {req.customer_email}")
    except Exception as exc:
        svc.memory.add_event(tid, "Email_Intake", "error", f"Acknowledgement email failed: {exc}")


async def mail_poller(svc: SupportServices) -> None:
    """Poll the support inbox for NEW mail only and turn genuine support requests into tickets.

    Newsletters, promotions, notifications, personal mail without a product/support keyword
    and anything that was already in the inbox when intake started are skipped (never
    ticketed, never answered, never marked as read). Replies quoting a ticket ID go to
    that ticket's chat.
    """
    from .emailer import fetch_new_mail, is_support_mail

    s = svc.settings
    while True:
        try:
            # Inbox position is tracked PER mailbox, so changing SUPPORT_MAILBOX starts fresh
            key = s.support_mailbox.lower()
            last = svc.memory.get_kv(f"mail_last_uid:{key}")
            validity = svc.memory.get_kv(f"mail_uidvalidity:{key}")
            mails, new_last, new_validity = await asyncio.to_thread(
                fetch_new_mail, s, int(last) if last is not None else None, validity)
            for m in mails:
                if svc.memory.mail_seen(m.message_id):
                    continue
                accept, reason = is_support_mail(m, s, svc.router.products)
                if not accept:
                    log.info("Mail skipped (%s): from=%s subject=%r", reason, m.from_email, m.subject[:60])
                    continue
                if m.existing_ticket and svc.memory.get_ticket(m.existing_ticket):
                    # Reply to one of our emails: YES/NO confirmation or a follow-up question
                    from .lifecycle import handle_customer_email

                    svc.memory.mark_mail(m.message_id, m.existing_ticket)
                    outcome = await asyncio.to_thread(handle_customer_email, svc, m.existing_ticket, m.body,
                                                      m.from_email)
                    log.info("Reply from %s on %s: %s", m.from_email, m.existing_ticket, outcome)
                    continue
                # Order ID only if the customer actually wrote one; otherwise left empty
                req = ComplaintRequest(customer_name=m.from_name[:120] or "Customer", order_id=m.order_id,
                                       complaint=(f"{m.subject}\n{m.body}")[:4000], customer_email=m.from_email,
                                       channel="email")
                tid = await submit_ticket(svc, req)
                svc.memory.mark_mail(m.message_id, tid)
                await asyncio.to_thread(send_ack, svc, tid, req, m.message_id)
                log.info("Support mail from %s accepted (%s) -> %s", m.from_email, reason, tid)
            svc.memory.set_kv(f"mail_last_uid:{key}", str(new_last))
            svc.memory.set_kv(f"mail_uidvalidity:{key}", new_validity)
        except Exception as exc:
            log.error("Mail polling error: %s", exc)
        await asyncio.sleep(s.mail_poll_seconds)


async def confirmation_sweeper(svc: SupportServices) -> None:
    """Close tickets whose customer has not confirmed within CONFIRMATION_TIMEOUT_HOURS."""
    from .lifecycle import close_stale

    while True:
        try:
            closed = await asyncio.to_thread(close_stale, svc)
            if closed:
                log.info("Auto-closed (no customer response): %s", ", ".join(closed))
        except Exception as exc:
            log.error("Confirmation sweep failed: %s", exc)
        await asyncio.sleep(svc.settings.confirmation_sweep_minutes * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _svc
    bootstrap_runtime()
    _svc = build_services()
    s = _svc.settings
    spawn(confirmation_sweeper(_svc))
    if s.email_enabled and s.support_mailbox and s.support_mailbox_password.get_secret_value():
        spawn(mail_poller(_svc))
        log.info("Mail intake enabled for %s", s.support_mailbox)
    yield
    for t in list(_tasks):
        t.cancel()


app = FastAPI(title="Customer Support Agent API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(MemoryStoreError)
async def storage_error_handler(request: Request, exc: MemoryStoreError) -> JSONResponse:
    """Database locked / unavailable: tell the client to retry instead of a raw 500."""
    log.error("Storage error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=503, headers={"Retry-After": "5"},
                        content={"detail": "Ticket storage is temporarily unavailable. Please try again shortly."})


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last line of defence: any unhandled error becomes a JSON 500 with a reference ID.
    The full traceback goes to the server log (and LangSmith); the client sees no internals."""
    error_id = secrets.token_hex(4).upper()
    log.exception("Unhandled error %s on %s %s", error_id, request.method, request.url.path)
    return JSONResponse(status_code=500, content={
        "detail": "Something went wrong on our side. Please try again; if it keeps happening, "
                  f"contact support with error ID {error_id}.", "error_id": error_id})


# ------------------------------- admin auth -------------------------------- #
def _secret() -> bytes:
    s = get_settings().admin_token_secret.get_secret_value()
    if not s:  # per-process secret if none configured (tokens reset on restart)
        global _EPHEMERAL
        try:
            return _EPHEMERAL
        except NameError:
            _EPHEMERAL = secrets.token_bytes(32)
            return _EPHEMERAL
    return s.encode()


def issue_token(username: str) -> str:
    payload = f"{username}|{int(time.time()) + _TOKEN_TTL}"
    sig = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}|{sig}".encode()).decode()


def verify_token(token: str) -> str:
    try:
        user, exp, sig = base64.urlsafe_b64decode(token.encode()).decode().split("|")
    except Exception:
        raise HTTPException(401, "Invalid token")
    good = hmac.new(_secret(), f"{user}|{exp}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(good, sig) or int(exp) < time.time():
        raise HTTPException(401, "Token expired or invalid")
    return user


def require_admin(authorization: str = Header(default="")) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Admin login required")
    return verify_token(authorization.removeprefix("Bearer ").strip())


@app.post("/admin/login")
def admin_login(body: AdminLogin) -> dict:
    s = get_settings()
    if not s.admin_password.get_secret_value():
        # No password in code: until ADMIN_PASSWORD is set in .env, nobody can sign in as admin
        raise HTTPException(503, "Admin login is disabled: set ADMIN_PASSWORD in the .env file")
    ok_user = hmac.compare_digest(body.username, s.admin_username)
    ok_pwd = hmac.compare_digest(body.password, s.admin_password.get_secret_value())
    if not (ok_user and ok_pwd):
        raise HTTPException(401, "Invalid credentials")
    return {"token": issue_token(body.username), "expires_in": _TOKEN_TTL}


@app.get("/admin/me")
def admin_me(user: str = Depends(require_admin)) -> dict:
    """Lets the UI check whether a saved admin token is still valid (used after a page refresh)."""
    return {"user": user}


# --------------------------------- public ---------------------------------- #
@app.get("/health")
def health(svc: SupportServices = Depends(services)) -> dict:
    return {"status": "ok", "products": [p["name"] for p in svc.router.products],
            "retrieval_mode": svc.settings.retrieval_mode.value}


@app.get("/products")
def products(svc: SupportServices = Depends(services)) -> list[dict]:
    return svc.router.products


@app.post("/tickets", response_model=TicketResponse, status_code=202)
async def create(req: ComplaintRequest, svc: SupportServices = Depends(services)) -> TicketResponse:
    tid = await submit_ticket(svc, req)
    return svc.memory.get_ticket(tid)


def ticket_access(ticket_id: str, code: str | None = Query(default=None),
                  order_id: str | None = Query(default=None),
                  authorization: str = Header(default=""),
                  svc: SupportServices = Depends(services)) -> TicketResponse:
    """A ticket is visible to an admin, or to whoever has its ticket ID plus EITHER its access code
    OR its order ID. Unknown ticket and wrong code return the same 404 so IDs cannot be probed."""
    t = svc.memory.get_ticket(ticket_id)
    if t is not None and authorization.startswith("Bearer "):
        verify_token(authorization.removeprefix("Bearer ").strip())
        return t
    given = (code or order_id or "").strip().lower()
    valid = {v.strip().lower() for v in (t.access_code, t.order_id) if v} if t else set()
    if not given or given not in valid:
        raise HTTPException(404, "No ticket matches this ticket ID and access code / order ID")
    return t


@app.get("/tickets/{ticket_id}", response_model=TicketResponse)
def get_ticket(t: TicketResponse = Depends(ticket_access)) -> TicketResponse:
    return t


@app.get("/tickets/{ticket_id}/events", response_model=list[AgentEvent])
def events(after: int = 0, t: TicketResponse = Depends(ticket_access),
           svc: SupportServices = Depends(services)) -> list[AgentEvent]:
    return svc.memory.list_events(t.ticket_id, after)


@app.get("/tickets/{ticket_id}/chat")
def chat_history(t: TicketResponse = Depends(ticket_access), svc: SupportServices = Depends(services)) -> list[dict]:
    return svc.memory.chat_history(t.ticket_id)


@app.post("/tickets/{ticket_id}/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, t: TicketResponse = Depends(ticket_access),
               svc: SupportServices = Depends(services)) -> ChatResponse:
    try:
        return await asyncio.to_thread(answer_followup, svc, t.ticket_id, body.message)
    except Exception:
        log.exception("chat failed")
        return ChatResponse(answer="Sorry, I couldn't answer right now – a support agent will follow up on "
                                   "your ticket.", sources=[], standalone_question=body.message)


@app.post("/tickets/{ticket_id}/confirm")
def confirm(body: ConfirmRequest, t: TicketResponse = Depends(ticket_access),
            svc: SupportServices = Depends(services)) -> dict:
    """Customer answers 'Did this solve your issue?' from the UI (ticket ID + order ID required)."""
    from .lifecycle import handle_confirmation

    result = handle_confirmation(svc, t.ticket_id, body.resolved, body.comment, "web")
    return {"result": result, "status": svc.memory.get_ticket(t.ticket_id).status}


# Ticket log: admin only (contains every customer's name, order and complaint)
@app.get("/admin/log")
def ticket_log(_: str = Depends(require_admin), svc: SupportServices = Depends(services)) -> list[dict]:
    return svc.logger.read_all()


@app.get("/admin/log.csv")
def ticket_log_csv(_: str = Depends(require_admin), svc: SupportServices = Depends(services)) -> StreamingResponse:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=TicketLogRow.columns(), extrasaction="ignore")
    w.writeheader()
    w.writerows(svc.logger.read_all())
    return StreamingResponse(iter([buf.getvalue().encode("utf-8-sig")]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=ticket_log.csv"})


# ---------------------------------- admin ---------------------------------- #
@app.get("/admin/dashboard")
def dashboard(_: str = Depends(require_admin), svc: SupportServices = Depends(services)) -> dict:
    tickets = svc.memory.list_tickets(5000)
    scored = [URGENCY_SCORE[t.urgency] for t in tickets if t.urgency in URGENCY_SCORE]
    status = Counter("Escalated" if t.status.startswith("Escalated") else t.status for t in tickets)
    return {
        "total": len(tickets),
        "per_category": dict(Counter(t.category or "unclassified" for t in tickets)),
        "status": dict(status),
        "per_product": dict(Counter(t.product or "unknown" for t in tickets)),
        "avg_urgency": round(sum(scored) / len(scored), 2) if scored else 0.0,
        "urgency_counts": dict(Counter(t.urgency for t in tickets if t.urgency)),
        "avg_confidence": round(sum(t.confidence or 0 for t in tickets) / len(tickets), 3) if tickets else 0.0,
        **_cost_summary(svc.memory.all_metrics()),
    }


def _cost_summary(metrics: list[dict]) -> dict:
    """Average processing time and approximate model cost per ticket (automated part only)."""
    if not metrics:
        return {"measured_tickets": 0, "avg_latency_s": None, "p90_latency_s": None,
                "avg_cost_usd": None, "total_cost_usd": None, "avg_tokens": None}
    lat = sorted(m.get("latency_s", 0) for m in metrics)
    cost = [m.get("cost_usd", 0) for m in metrics]
    tokens = [m.get("tokens_in", 0) + m.get("tokens_out", 0) for m in metrics]
    return {"measured_tickets": len(metrics),
            "avg_latency_s": round(sum(lat) / len(lat), 1),
            "p90_latency_s": round(lat[min(len(lat) - 1, int(0.9 * len(lat)))], 1),
            "avg_cost_usd": round(sum(cost) / len(cost), 5),
            "total_cost_usd": round(sum(cost), 4),
            "avg_tokens": round(sum(tokens) / len(tokens))}


@app.get("/admin/approvals", response_model=list[TicketResponse])
def approvals(_: str = Depends(require_admin), svc: SupportServices = Depends(services)) -> list[TicketResponse]:
    return [t for tid in svc.memory.pending_approvals() if (t := svc.memory.get_ticket(tid))]


@app.post("/admin/approvals/{ticket_id}")
def decide(ticket_id: str, body: ApprovalRequest, user: str = Depends(require_admin),
           svc: SupportServices = Depends(services)) -> dict:
    if body.decision == "edit" and not (body.edited_reply or "").strip():
        raise HTTPException(422, "edited_reply is required for an edit")
    if not svc.memory.decide_approval(ticket_id, body.decision, body.approver or user, body.edited_reply, body.note):
        raise HTTPException(409, "No pending approval for this ticket")
    # Apply the decision right away (send the email, update the log) so the admin sees the
    # delivery result. The running agent conversation's UserProxyAgent (Support_Approver)
    # still receives the same decision and the Log_Agent records it before LOGGED.
    result = finalize_after_pipeline(svc, ticket_id)
    return {"ok": True, "detail": result or "decision recorded; the agent team is applying it"}


@app.get("/admin/escalations", response_model=list[TicketResponse])
def escalations(_: str = Depends(require_admin), svc: SupportServices = Depends(services)) -> list[TicketResponse]:
    """Escalated tickets that support is handling directly (no pending reply approval)."""
    return svc.memory.tickets_with_status("Escalated")


@app.post("/admin/tickets/{ticket_id}/resolve")
def resolve(ticket_id: str, body: ResolveRequest, user: str = Depends(require_admin),
            svc: SupportServices = Depends(services)) -> dict:
    """Support finishes a ticket it handled: send resolution for confirmation, or close now."""
    from .lifecycle import admin_resolve

    try:
        detail = admin_resolve(svc, ticket_id, body.mode, body.note, body.notify_customer, body.resolver or user)
    except KeyError:
        raise HTTPException(404, "Ticket not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {"ok": True, "detail": detail, "status": svc.memory.get_ticket(ticket_id).status}


@app.get("/admin/tickets/{ticket_id}/retrieval")
def ticket_retrieval(ticket_id: str, _: str = Depends(require_admin), svc: SupportServices = Depends(services)) -> dict:
    extra = svc.memory.get_extra(ticket_id)
    return {k: extra.get(k) for k in ("analysis", "passages", "graph_facts", "retrieval_mode", "retrieval_errors",
                                      "decision", "draft", "metrics")}


@app.get("/admin/chunks")
def chunks(product: str, q: str | None = None, k: int = Query(10, le=50), _: str = Depends(require_admin),
           svc: SupportServices = Depends(services)) -> list[dict]:
    from .retrieval import bm25_search, vector_search

    try:
        idx = svc.registry.get(product)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    if q:
        ids = list(dict.fromkeys(vector_search(idx, q, k) + bm25_search(idx, q, k)))[:k]
    else:
        ids = idx.order[:k]
    return [idx.chunks[i].model_dump() for i in ids]


@app.get("/admin/graph")
def graph(product: str, limit: int = Query(50, le=300), _: str = Depends(require_admin),
          svc: SupportServices = Depends(services)) -> list[dict]:
    from .graph import get_driver, graph_sample

    try:
        with get_driver(svc.settings) as drv:
            return graph_sample(drv, product, svc.settings, limit)
    except Exception as exc:
        raise HTTPException(503, f"Knowledge graph unavailable: {exc}")


@app.post("/admin/reload")
def reload(_: str = Depends(require_admin), svc: SupportServices = Depends(services)) -> dict:
    svc.reload_indexes()
    return {"ok": True, "products": [p["name"] for p in svc.router.products]}
