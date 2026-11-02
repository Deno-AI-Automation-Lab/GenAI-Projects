"""Customer-confirmation lifecycle: neutral subjects, YES/NO handling, re-escalation, auto-close."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from support_agent.emailer import _text_body, strip_markdown, subject_title
from support_agent.lifecycle import classify_rules, close_stale, handle_confirmation, handle_customer_email
from support_agent.models import local_tz
from support_agent.services import build_services


def test_subject_is_neutral_and_short():
    t = subject_title("AeroBeat X9 Wireless Noise-Cancelling Headphones", "Headphones not pairing", "technical_issue")
    assert t == "AeroBeat X9 - Headphones not pairing"
    assert subject_title("SmartCrisp Pro 8L Wi-Fi Air Fryer", "", "refund") == "SmartCrisp Pro - Refund request"
    assert len(subject_title("X", "word " * 40, None)) <= 60


def test_markdown_removed_from_email():
    assert strip_markdown("1. **Pairing mode**: hold `Power`") == "1. Pairing mode: hold Power"


@pytest.mark.parametrize("text,intent", [
    ("Yes, it works now. Thanks!", "confirmed"), ("YES", "confirmed"), ("Solved, thank you", "confirmed"),
    ("No, it is still not working", "not_resolved"), ("nope", "not_resolved"),
    ("I tried that but it didn't work", "not_resolved"), ("ஆம்", "confirmed"), ("இல்லை, இன்னும் வேலை செய்யவில்லை", "not_resolved"),
    ("Can I also use it with my TV?", None),
])
def test_reply_rules(text, intent):
    assert classify_rules(text) == intent


def test_gmail_wrapped_quote_is_removed():
    import email
    raw = ("From: a@b.c\r\nSubject: Re: x\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n"
           "Yes it works\r\n\r\nOn Mon, Nov 2, 2026 at 5:37 PM Theofaust Ventures Support <\r\n"
           "support@x.com> wrote:\r\n\r\n> Dear Deno\r\n> Did this solve your issue?\r\n")
    assert _text_body(email.message_from_string(raw)) == "Yes it works"


def _pending(svc, email_addr="cust@example.com"):
    tid = svc.memory.create_ticket(customer_name="Ravi", order_id="ORD-1", complaint="won't pair",
                                   customer_email=email_addr, channel="email")
    svc.memory.update_ticket(tid, status="Pending Confirmation", confirmation_requested_at=datetime.now(local_tz()).isoformat(),
                             extra={"analysis": {"product_name": "AeroBeat X9 Headphones", "issue_title": "Not pairing",
                                                 "language": "en"}, "draft": {"body": "Hold **Power** 3 s"}})
    return tid


def test_customer_confirms_via_ui(settings):
    svc = build_services(settings)
    tid = _pending(svc)
    out = handle_confirmation(svc, tid, True, None, "web")
    assert out.startswith("resolved") and svc.memory.get_ticket(tid).status == "Resolved"
    assert handle_confirmation(svc, tid, False, "x", "web").startswith("no change")   # idempotent
    assert svc.logger.read_csv()[-1]["status"] == "Resolved"


def test_customer_rejects_via_email_escalates_high(settings):
    svc = build_services(settings)
    tid = _pending(svc)
    out = handle_customer_email(svc, tid, "No, still not working after the reset", "CUST@example.com")
    t = svc.memory.get_ticket(tid)
    assert out.startswith("escalated") and t.status == "Escalated - Awaiting Approval" and t.urgency == "High"
    assert any("HIGH priority" in r for r in t.escalation_reasons)
    assert tid in svc.memory.pending_approvals()
    assert t.reply_body == "Hold Power 3 s"                      # approver edits a clean draft


def test_reply_from_other_sender_is_ignored(settings):
    svc = build_services(settings)
    tid = _pending(svc)
    assert handle_customer_email(svc, tid, "yes", "someone@else.com").startswith("ignored")
    assert svc.memory.get_ticket(tid).status == "Pending Confirmation"


def test_auto_close_after_timeout_and_reopen_on_reply(settings):
    svc = build_services(settings)
    tid = _pending(svc)
    assert close_stale(svc) == []                                 # too early
    later = datetime.now(local_tz()) + timedelta(hours=settings.confirmation_timeout_hours + 1)
    assert close_stale(svc, now=later) == [tid]
    assert svc.memory.get_ticket(tid).status == "Closed - No Response"
    handle_customer_email(svc, tid, "Yes it's fine now", "cust@example.com")    # reply after close
    assert svc.memory.get_ticket(tid).status == "Resolved"


# ----------------------- support closes an escalation ------------------- #
def _escalated(svc, status="Escalated"):
    tid = _pending(svc)
    handle_confirmation(svc, tid, False, "still broken", "web")      # -> Escalated - Awaiting Approval (High)
    if status == "Escalated":                                          # support chose "Handle it myself"
        from support_agent.agents import TicketContext, apply_decision
        svc.memory.decide_approval(tid, "reject", "admin", None, "calling the customer")
        t = svc.memory.get_ticket(tid)
        apply_decision(TicketContext(services=svc, ticket_id=tid, customer_name=t.customer_name, order_id=t.order_id,
                                     complaint=t.complaint, customer_email=t.customer_email), "reject", "calling")
    return tid


def test_support_sends_resolution_for_confirmation(settings):
    from support_agent.lifecycle import admin_resolve
    svc = build_services(settings)
    tid = _escalated(svc)
    assert svc.memory.get_ticket(tid).status == "Escalated"
    with pytest.raises(ValueError):
        admin_resolve(svc, tid, "confirm", "   ")                       # resolution text required
    out = admin_resolve(svc, tid, "confirm", "We replaced the cable; it pairs now.")
    t = svc.memory.get_ticket(tid)
    assert out.startswith("resolution sent") and t.status == "Pending Confirmation"
    handle_confirmation(svc, tid, True, None, "web")                     # customer confirms -> closed
    assert svc.memory.get_ticket(tid).status == "Resolved"


def test_support_closes_directly_and_supersedes_pending_approval(settings):
    from support_agent.lifecycle import admin_resolve
    svc = build_services(settings)
    tid = _escalated(svc, status="Escalated - Awaiting Approval")       # still in the Approvals queue
    out = admin_resolve(svc, tid, "close", "Fixed over a phone call.")
    assert out.startswith("resolved and closed") and svc.memory.get_ticket(tid).status == "Resolved"
    assert tid not in svc.memory.pending_approvals()
    assert svc.logger.read_csv()[-1]["status"] == "Resolved"
    assert admin_resolve(svc, tid, "close", "").startswith("no change")


# ------------------------- optional order ID ---------------------------- #
def test_order_id_optional_blank_not_placeholder(settings):
    from support_agent.emailer import ref_line, render_ack_email
    from support_agent.models import ComplaintRequest
    assert ComplaintRequest(customer_name="A", complaint="Vacuum is slow", order_id="  ").order_id is None
    svc = build_services(settings)
    tid = svc.memory.create_ticket(customer_name="A", order_id=None, complaint="slow", customer_email=None, channel="email")
    t = svc.memory.get_ticket(tid)
    assert t.order_id is None and len(t.access_code) == 6
    from support_agent.lifecycle import log_row
    assert log_row(t, "Pending Confirmation", "x").order_id == ""          # blank cell in the sheet
    assert "Order ID" not in ref_line(tid, None, "Open")
    subj, body = render_ack_email(ticket_id=tid, customer_name="A", order_id=None, access_code=t.access_code,
                                  site_url="http://x", company="C")
    assert "Order ID" not in body and "EMAIL-" not in body and t.access_code in body


def test_draft_asks_for_order_only_when_needed(settings):
    from support_agent.models import Passage, QueryAnalysis, RetrievalResult
    from support_agent.resolution import draft_reply
    seen = []

    class FakeLLM:
        def invoke(self, msgs):
            seen.append(msgs[-1].content)
            class R: content = "Answer [S1]"
            return R()
    res = RetrievalResult(product="p", query="q", context="[S1] text", mode="hybrid",
                          passages=[Passage(chunk_id="c", source_file="f.docx", section_title="s", text="t")])
    base = dict(standalone_question="q", product="p", urgency="Low", sentiment="Neutral", summary="s", fusion_queries=["q"])
    draft_reply(QueryAnalysis(category="technical_issue", **base), res, settings, llm=FakeLLM(), order_id=None)
    draft_reply(QueryAnalysis(category="refund", **base), res, settings, llm=FakeLLM(), order_id=None)
    assert "order number" not in seen[0] and "order number" in seen[1]


def test_log_timestamp_format(settings):
    import re
    from support_agent.lifecycle import log_row
    svc = build_services(settings)
    tid = svc.memory.create_ticket(customer_name="A", order_id=None, complaint="x", customer_email=None, channel="web")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", log_row(svc.memory.get_ticket(tid), "Resolved", "x").timestamp)
