"""Ticket lifecycle after a reply is sent: customer confirmation, re-escalation, auto-close.

    reply sent ──> Pending Confirmation ──(customer: yes)──> Resolved
                                        ──(customer: no)───> Escalated - Awaiting Approval (High priority)
                                        ──(no answer in CONFIRMATION_TIMEOUT_HOURS)──> Closed - No Response

Customers answer by replying to the email (YES / NO + details) or with the buttons in
'My tickets' on the support site. Email replies are only accepted from the address the
ticket belongs to. Free-text replies are classified by rules first, then an LLM.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from .emailer import (TAG_CLOSED, TAG_PROPOSED, TAG_REOPENED, TAG_RESOLVED, build_subject, confirmation_block,
                      render_customer_email, send_email, subject_title)
from .models import (CLOSED_NO_RESPONSE, CLOSED_UNSUPPORTED, ESCALATED_WAIT, PENDING, RESOLVED, CustomerReplyIntent, TicketLogRow,
                     TicketResponse, local_tz, log_time, now_iso)

log = logging.getLogger(__name__)

AGENT = "Customer_Feedback"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _analysis(svc, ticket_id: str) -> dict:
    return (svc.memory.get_extra(ticket_id) or {}).get("analysis") or {}


def ticket_subject_title(svc, t: TicketResponse) -> str:
    a = _analysis(svc, t.ticket_id)
    return subject_title(a.get("product_name"), a.get("issue_title"), t.category)


def log_row(t: TicketResponse, status: str, response: str) -> TicketLogRow:
    """Ticket-log row built from the stored ticket (same 12 columns as the Log Agent writes)."""
    return TicketLogRow(
        ticket_id=t.ticket_id, timestamp=log_time(), customer_name=t.customer_name, order_id=t.order_id or "",
        complaint=t.complaint, category=t.category or "", urgency=t.urgency or "", sentiment=t.sentiment or "",
        kb_sources="; ".join(t.kb_sources), response=response, status=status,
        escalation_reason="; ".join(t.escalation_reasons) if status.startswith("Escalated") else "",
    )


def _send(svc, t: TicketResponse, subject: str, body: str) -> str:
    """Send to the customer if an email is on file; always keep the message in the ticket chat."""
    svc.memory.add_chat(t.ticket_id, "assistant", body)
    if not t.customer_email:
        return "no customer email on file (shown in UI only)"
    try:
        send_email(svc.settings, t.customer_email, subject, body)
        svc.memory.add_event(t.ticket_id, AGENT, "system", f"Email sent: {subject}")
        return f"emailed to {t.customer_email}"
    except Exception as exc:
        log.error("%s: email to %s failed: %s", t.ticket_id, t.customer_email, exc)
        svc.memory.add_event(t.ticket_id, AGENT, "error", f"Email failed: {exc}")
        return f"email failed ({exc.__class__.__name__}: {exc})"


def deliver_reply(svc, ticket_id: str, *, tag: str, body: str, sources: list[str], escalated: bool) -> str:
    """Send a proposed resolution with the standard template + confirmation instructions."""
    t = svc.memory.get_ticket(ticket_id)
    a = _analysis(svc, ticket_id)
    s = svc.settings
    subject = build_subject(tag, ticket_id, ticket_subject_title(svc, t))
    full = render_customer_email(
        ticket_id=ticket_id, status="Awaiting your confirmation", customer_name=t.customer_name,
        order_id=t.order_id, body=body, sources=sources, language=a.get("language", "en"),
        company=s.company_name, escalated=escalated,
        confirm_lines=confirmation_block(has_email=bool(t.customer_email), site_url=s.support_site_url,
                                         timeout_hours=s.confirmation_timeout_hours, access_code=t.access_code))
    svc.memory.update_ticket(ticket_id, reply_subject=subject, reply_body=full)
    return _send(svc, t, subject, full)


def close_unsupported(svc, ticket_id: str, body: str) -> str:
    """Send the 'we only support these products' reply (no YES/NO) and close the ticket."""
    from .emailer import TAG_CLOSED

    t = svc.memory.get_ticket(ticket_id)
    a = _analysis(svc, ticket_id)
    subject = build_subject(TAG_CLOSED, ticket_id, "Product not supported")
    full = render_customer_email(
        ticket_id=ticket_id, status="Closed", customer_name=t.customer_name, order_id=t.order_id, body=body,
        sources=[], language=a.get("language", "en"), company=svc.settings.company_name,
        has_email=bool(t.customer_email))
    svc.memory.update_ticket(ticket_id, status=CLOSED_UNSUPPORTED, reply_subject=subject, reply_body=full)
    t = svc.memory.get_ticket(ticket_id)
    svc.logger.upsert(log_row(t, CLOSED_UNSUPPORTED, body))
    svc.memory.add_event(ticket_id, AGENT, "system", "Product not supported - customer informed, ticket closed")
    return _send(svc, t, subject, full)


def request_confirmation(svc, ticket_id: str, response: str) -> None:
    """Move the ticket to Pending Confirmation and log it."""
    svc.memory.update_ticket(ticket_id, status=PENDING, confirmation_requested_at=now_iso())
    t = svc.memory.get_ticket(ticket_id)
    svc.logger.upsert(log_row(t, PENDING, response))
    svc.memory.add_event(ticket_id, AGENT, "system",
                         f"Waiting for the customer to confirm the resolution "
                         f"(auto-close after {svc.settings.confirmation_timeout_hours} h)")


# --------------------------------------------------------------------------- #
# Customer answer
# --------------------------------------------------------------------------- #
_NEGATIVE = re.compile(
    r"\b(not (?:working|solved|resolved|fixed|helpful)|still (?:not|the same|broken|failing|doesn'?t)|"
    r"didn'?t (?:work|help|solve)|doesn'?t (?:work|help)|no luck|same (?:issue|problem)|not yet)\b|இல்லை|नहीं", re.I)
_YES_WORDS = {"yes", "yeah", "yep", "yup", "y", "ok", "okay", "solved", "resolved", "fixed", "done", "works",
              "ஆம்", "சரி", "हाँ", "हां", "जी"}
_NO_WORDS = {"no", "nope", "nah", "n"}
_POSITIVE = re.compile(r"\b(solved|resolved|fixed|it works|working now|works now|thank(?:s| you))\b", re.I)


def classify_rules(text: str) -> str | None:
    """Fast deterministic intent: 'confirmed' / 'not_resolved' / None (unclear)."""
    head = (text or "").strip()[:300]
    if not head:
        return None
    if _NEGATIVE.search(head):
        return "not_resolved"
    first = re.split(r"[\s,.!;:-]+", head.lower(), maxsplit=1)[0]
    if first in _NO_WORDS:
        return "not_resolved"
    if first in _YES_WORDS:
        return "confirmed"
    if _POSITIVE.search(head):
        return "confirmed"
    return None


@traceable(name="classify_customer_reply")
def classify_customer_reply(text: str, llm=None) -> str:
    """Rules first; LLM (structured output) for anything unclear or non-English."""
    intent = classify_rules(text)
    if intent:
        return intent
    try:
        from .llm import get_chat_llm

        res: CustomerReplyIntent = (llm or get_chat_llm()).with_structured_output(CustomerReplyIntent).invoke([
            SystemMessage(content="A support team sent a customer a proposed solution and asked whether it "
                                  "solved the issue. Classify the customer's reply: confirmed (it is solved), "
                                  "not_resolved (still has the problem or is unhappy with the solution), or "
                                  "question (asks something else / unclear)."),
            HumanMessage(content=text[:2000]),
        ])
        return res.intent
    except Exception as exc:
        log.warning("Reply classification fell back to 'question': %s", exc)
        return "question"


def handle_confirmation(svc, ticket_id: str, resolved: bool, comment: str | None, source: str) -> str:
    """Apply the customer's yes/no. Idempotent: only acts while the ticket is Pending Confirmation."""
    t = svc.memory.get_ticket(ticket_id)
    if t is None:
        raise KeyError(ticket_id)
    if t.status != PENDING:
        return f"no change: ticket is '{t.status}'"
    comment = (comment or "").strip()
    svc.memory.update_ticket(ticket_id, customer_feedback=f"{'confirmed' if resolved else 'not resolved'}"
                                                          f" via {source}: {comment}"[:2000])
    if comment:
        svc.memory.add_chat(ticket_id, "user", comment)
    title = ticket_subject_title(svc, t)
    a = _analysis(svc, ticket_id)
    s = svc.settings

    if resolved:
        svc.memory.update_ticket(ticket_id, status=RESOLVED)
        t = svc.memory.get_ticket(ticket_id)
        svc.logger.upsert(log_row(t, RESOLVED, f"Customer confirmed the resolution ({source})."))
        svc.memory.add_event(ticket_id, AGENT, "system", f"Customer confirmed the resolution via {source} - Resolved")
        body = render_customer_email(
            ticket_id=ticket_id, status="Resolved", customer_name=t.customer_name, order_id=t.order_id,
            body="Thank you for confirming that your issue is solved. We have closed this ticket. "
                 "If anything else comes up, just reply to this email or raise a new ticket.",
            sources=[], language=a.get("language", "en"), company=s.company_name, has_email=bool(t.customer_email))
        delivery = _send(svc, t, build_subject(TAG_RESOLVED, ticket_id, title), body)
        return f"resolved; {delivery}"

    # Not resolved -> high-priority escalation for a human
    reasons = list(t.escalation_reasons) + [
        "Customer reported the proposed resolution did not solve the issue (HIGH priority)"
        + (f": {comment[:300]}" if comment else "")]
    draft = ((svc.memory.get_extra(ticket_id) or {}).get("draft") or {}).get("body") or ""
    from .emailer import strip_markdown

    svc.memory.update_ticket(ticket_id, status=ESCALATED_WAIT, urgency="High", escalation_reasons=reasons,
                             reply_body=strip_markdown(draft))   # approver edits a clean draft
    svc.memory.request_approval(ticket_id)
    t = svc.memory.get_ticket(ticket_id)
    svc.logger.upsert(log_row(t, ESCALATED_WAIT, f"Customer said not resolved ({source}): {comment}"))
    svc.memory.add_event(ticket_id, AGENT, "system",
                         f"Customer said the issue is NOT solved via {source} - escalated as High priority")
    from .emailer import render_escalation_email

    subj, mail = render_escalation_email(
        ticket_id=ticket_id, customer_name=t.customer_name, customer_email=t.customer_email,
        order_id=t.order_id, complaint=t.complaint + (f"\n\nCustomer feedback on our reply:\n{comment}" if comment else ""),
        analysis=a | {"urgency": "High"}, reasons=reasons, draft_reply=t.reply_body or "", sources=t.kb_sources)
    try:
        send_email(s, s.escalation_email, "[HIGH PRIORITY] " + subj, mail)
    except Exception as exc:
        svc.memory.add_event(ticket_id, AGENT, "error", f"Escalation email failed: {exc}")
    body = render_customer_email(
        ticket_id=ticket_id, status="Escalated - High priority", customer_name=t.customer_name,
        order_id=t.order_id,
        body="We're sorry the suggested solution did not fix the problem. Your ticket has been escalated to a "
             "senior support specialist as high priority, and they will contact you shortly.",
        sources=[], language=a.get("language", "en"), company=s.company_name, has_email=bool(t.customer_email))
    delivery = _send(svc, t, build_subject(TAG_REOPENED, ticket_id, title), body)
    return f"escalated as high priority; {delivery}"


def handle_customer_email(svc, ticket_id: str, body: str, from_email: str) -> str:
    """Email reply quoting a ticket ID: confirmation, rejection, or a follow-up question."""
    t = svc.memory.get_ticket(ticket_id)
    if t is None:
        return "unknown ticket"
    if not t.customer_email or t.customer_email.lower() != (from_email or "").lower():
        svc.memory.add_event(ticket_id, AGENT, "system", f"Ignored a reply from {from_email} (not the ticket owner)")
        return "ignored: sender is not the ticket owner"
    if t.status == CLOSED_NO_RESPONSE:
        # The closing email invites a reply: reopen and treat it like an answer to the proposal
        svc.memory.update_ticket(ticket_id, status=PENDING)
        svc.memory.add_event(ticket_id, AGENT, "system", "Customer replied after auto-close - ticket reopened")
        t = svc.memory.get_ticket(ticket_id)
    if t.status != PENDING:
        svc.memory.add_chat(ticket_id, "user", f"(email) {body}")
        svc.memory.add_event(ticket_id, AGENT, "system", "Customer replied by email")
        return f"added to chat (ticket is '{t.status}')"
    intent = classify_customer_reply(body)
    svc.memory.add_event(ticket_id, AGENT, "system", f"Customer email reply classified as: {intent}")
    if intent in ("confirmed", "not_resolved"):
        return handle_confirmation(svc, ticket_id, intent == "confirmed", body, "email")
    # A follow-up question: answer from the documents, keep waiting for confirmation
    from .pipeline import answer_followup

    try:
        ans = answer_followup(svc, ticket_id, body)
        a = _analysis(svc, ticket_id)
        mail = render_customer_email(
            ticket_id=ticket_id, status="Awaiting your confirmation", customer_name=t.customer_name,
            order_id=t.order_id, body=ans.answer, sources=ans.sources, language=a.get("language", "en"),
            company=svc.settings.company_name,
            confirm_lines=confirmation_block(has_email=True, site_url=svc.settings.support_site_url,
                                             timeout_hours=svc.settings.confirmation_timeout_hours,
                                             access_code=t.access_code))
        return "answered follow-up; " + _send(svc, t, build_subject(TAG_PROPOSED, ticket_id,
                                                                    ticket_subject_title(svc, t)), mail)
    except Exception as exc:
        log.error("Follow-up answer failed for %s: %s", ticket_id, exc)
        svc.memory.add_chat(ticket_id, "user", f"(email) {body}")
        return f"follow-up stored; answer failed ({exc})"


# --------------------------------------------------------------------------- #
# Auto-close
# --------------------------------------------------------------------------- #
def _parse(ts: str | None) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts) if ts else None
        return dt if dt is None or dt.tzinfo else dt.replace(tzinfo=local_tz())
    except ValueError:
        return None


def close_stale(svc, now: datetime | None = None) -> list[str]:
    """Close tickets that have waited longer than CONFIRMATION_TIMEOUT_HOURS for the customer."""
    now = now or datetime.now(local_tz())
    limit = timedelta(hours=svc.settings.confirmation_timeout_hours)
    closed = []
    for t in svc.memory.tickets_with_status(PENDING):
        since = _parse(t.confirmation_requested_at) or _parse(t.updated_at)
        if since is None or now - since < limit:
            continue
        svc.memory.update_ticket(t.ticket_id, status=CLOSED_NO_RESPONSE)
        t2 = svc.memory.get_ticket(t.ticket_id)
        svc.logger.upsert(log_row(t2, CLOSED_NO_RESPONSE, "No response from the customer - closed automatically."))
        svc.memory.add_event(t.ticket_id, AGENT, "system", "No response from the customer - ticket closed")
        a = _analysis(svc, t.ticket_id)
        body = render_customer_email(
            ticket_id=t.ticket_id, status="Closed", customer_name=t.customer_name, order_id=t.order_id,
            body="We haven't heard back from you, so we have closed this ticket. If the issue is still not "
                 "solved, simply reply to this email and we will reopen it for you.",
            sources=[], language=a.get("language", "en"), company=svc.settings.company_name)
        _send(svc, t2, build_subject(TAG_CLOSED, t.ticket_id, ticket_subject_title(svc, t2)), body)
        closed.append(t.ticket_id)
    return closed


# --------------------------------------------------------------------------- #
# Support closes an escalation it handled itself
# --------------------------------------------------------------------------- #
FINAL_STATUSES = (RESOLVED, CLOSED_NO_RESPONSE, CLOSED_UNSUPPORTED)


def admin_resolve(svc, ticket_id: str, mode: str, note: str, notify_customer: bool = True,
                  resolver: str = "support") -> str:
    """Finish an escalated ticket that support handled directly (phone call, replacement, ...).

    mode='confirm': send the resolution note with the YES/NO request -> Pending Confirmation
                    (the normal confirmation loop and 2-day auto-close apply again).
    mode='close'  : mark Resolved immediately (optional closing email with the note).
    """
    t = svc.memory.get_ticket(ticket_id)
    if t is None:
        raise KeyError(ticket_id)
    if t.status in FINAL_STATUSES:
        return f"no change: ticket is already '{t.status}'"
    note = (note or "").strip()
    if mode == "confirm" and not note:
        raise ValueError("Write the resolution to send to the customer")
    # Any pending approval is superseded by this decision (prevents a second, conflicting reply)
    appr = svc.memory.get_approval(ticket_id)
    if appr and appr.get("state") == "pending":
        svc.memory.decide_approval(ticket_id, "reject", resolver, None, f"handled by support ({mode})")
        svc.memory.claim_approval(ticket_id)
    svc.memory.add_event(ticket_id, "Support_Team", "system",
                         f"{resolver} {'sent a resolution for confirmation' if mode == 'confirm' else 'closed the ticket'}"
                         + (f": {note[:200]}" if note else ""))
    if mode == "confirm":
        from .emailer import TAG_ESCALATED

        delivery = deliver_reply(svc, ticket_id, tag=TAG_ESCALATED, body=note, sources=[], escalated=True)
        request_confirmation(svc, ticket_id, note)
        return f"resolution sent, waiting for the customer to confirm; {delivery}"

    svc.memory.update_ticket(ticket_id, status=RESOLVED,
                             customer_feedback=f"closed by {resolver}" + (f": {note[:500]}" if note else ""))
    t = svc.memory.get_ticket(ticket_id)
    svc.logger.upsert(log_row(t, RESOLVED, note or f"Resolved by {resolver}."))
    delivery = "customer not notified"
    if notify_customer:
        a = _analysis(svc, ticket_id)
        body = render_customer_email(
            ticket_id=ticket_id, status="Resolved", customer_name=t.customer_name, order_id=t.order_id,
            body=(note + "\n\n" if note else "") + "We have now closed this ticket. If you need anything else, "
                 "just reply and we will be happy to help.",
            sources=[], language=a.get("language", "en"), company=svc.settings.company_name,
            escalated=True, has_email=bool(t.customer_email))
        delivery = _send(svc, t, build_subject(TAG_RESOLVED, ticket_id, ticket_subject_title(svc, t)), body)
    return f"resolved and closed; {delivery}"
