"""AutoGen wiring test with a scripted (replay) model client – no OpenAI calls.

Verifies: SelectorGroupChat order Query → Knowledge → Resolution → Log, tools invoked,
UserProxyAgent approval, TextMentionTermination on LOGGED, and model client closed.
"""
from __future__ import annotations

import asyncio
import threading
import time

from autogen_core import FunctionCall
from autogen_core.models import CreateResult, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient

from support_agent import agents
from support_agent.config import product_slug
from support_agent.ingest import build_router_index, discover_products, ingest_product
from support_agent.models import ComplaintRequest
from support_agent.pipeline import create_ticket, process_ticket
from support_agent.planner import load_router
from support_agent.services import build_services

FRYER = product_slug("SmartCrisp Pro 8L Wi-Fi Air Fryer")
U = RequestUsage(prompt_tokens=0, completion_tokens=0)


def _call(name: str, args: str, i: int) -> CreateResult:
    return CreateResult(finish_reason="function_calls", usage=U, cached=False,
                        content=[FunctionCall(id=f"c{i}", name=name, arguments=args)])


def _text(t: str) -> CreateResult:
    return CreateResult(finish_reason="stop", content=t, usage=U, cached=False)


class TrackingReplay(ReplayChatCompletionClient):
    closed = False

    async def close(self) -> None:
        TrackingReplay.closed = True
        await super().close()


def test_team_runs_in_order_with_human_approval(settings, fake_embeddings, monkeypatch):
    products = discover_products(settings.data_dir)
    for p in products:
        ingest_product(p, settings, embeddings=fake_embeddings, build_graph=False)
    build_router_index(products, settings, fake_embeddings)
    s = settings.model_copy(update={"approval_timeout_s": 20})
    svc = build_services(s)
    svc.registry.embeddings = fake_embeddings
    svc._router = load_router(s, fake_embeddings)
    monkeypatch.setattr(svc.settings, "openai_api_key", type(s.openai_api_key)("sk-test"))

    script = [
        _call("analyze_complaint", '{"complaint_text": "x"}', 1), _text("Product: smartcrisp. Category: technical_issue."),
        _call("search_knowledge_base", f'{{"query": "error E3", "product": "{FRYER}"}}', 2), _text("[product_faq.docx] E3 = basket."),
        _call("draft_customer_reply", '{"summary": "E3 error"}', 3), _text("Draft ready. Confidence: 0.00"),
        _call("record_ticket", '{"note": ""}', 4), _text("Escalated: low confidence. AWAITING_APPROVAL"),
        _call("finalize_ticket", '{"approver_message": "APPROVE"}', 5), _text("Approved reply sent. LOGGED"),
    ]
    client = TrackingReplay(script, model_info={"vision": False, "function_calling": True, "json_output": False,
                                                "family": "unknown", "structured_output": False})
    monkeypatch.setattr(agents, "build_model_client", lambda _s: client)

    ctx = create_ticket(svc, ComplaintRequest(customer_name="Ravi", order_id="ORD-10001",
                                              complaint="My SmartCrisp air fryer shows error E3 every time"))

    def approve_later():  # simulates the support person clicking "Approve" in the admin console
        for _ in range(100):
            if ctx.ticket_id in svc.memory.pending_approvals():
                svc.memory.decide_approval(ctx.ticket_id, "approve", "tester", None, None)
                return
            time.sleep(0.1)

    threading.Thread(target=approve_later, daemon=True).start()
    status = asyncio.run(process_ticket(svc, ctx))

    events = svc.memory.list_events(ctx.ticket_id)
    speakers = [e.agent for e in events if e.kind == "message"]
    order = [a for a in dict.fromkeys(speakers) if a.endswith("Agent")]
    assert order == ["Customer_Query_Agent", "Knowledge_Base_Agent", "Resolution_Agent", "Log_Agent"]
    assert "Support_Approver" in speakers
    assert any("LOGGED" in e.content for e in events if e.agent == "Log_Agent")
    assert any("Conversation stopped" in e.content and "LOGGED" in e.content for e in events)
    assert status == "Pending Confirmation"            # approved reply sent, waiting for the customer
    assert svc.memory.get_ticket(ctx.ticket_id).reply_subject.startswith("[Escalated] Ticket")
    assert TrackingReplay.closed                       # model client closed at the end
