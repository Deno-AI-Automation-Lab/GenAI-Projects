"""Brief scenarios end to end, offline, with recorded LLM replies (no OpenAI, Neo4j or Gmail).

Each case in brief_cases.json runs the real workflow (create_ticket -> process_ticket: Query,
Knowledge, Resolution and Log steps, escalation rules, approval wait) on the sample product
documents. Only the model calls are replaced by RecordedLLM, which returns the answers a
real model gave for that complaint (query plan, grounded answer, judge scores).
So the test checks our own logic - routing, retrieval, guardrails, escalation rules and
ticket status - deterministically, for every escalation rule in the brief.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from support_agent import llm as llm_module
from support_agent.config import product_slug
from support_agent.ingest import build_router_index, discover_products, ingest_product
from support_agent.models import ComplaintRequest, JudgeVerdict, QueryPlan
from support_agent.pipeline import create_ticket, process_ticket
from support_agent.planner import load_router
from support_agent.services import build_services

CASES = json.loads((Path(__file__).parent / "brief_cases.json").read_text(encoding="utf-8"))


def _slug(name: str) -> str:
    return name if name == "unknown" else product_slug(name)


class RecordedLLM:
    """Stands in for ChatOpenAI: returns the recorded reply for each kind of call."""

    def __init__(self, recorded: dict):
        plan = dict(recorded["plan"], product=_slug(recorded["plan"]["product"]))
        self._structured = {QueryPlan: QueryPlan(**plan), JudgeVerdict: JudgeVerdict(**recorded["judge"])}
        self._text = recorded["answer"]
        self.calls: list[str] = []

    def with_structured_output(self, schema):
        llm = self

        class _Structured:
            def invoke(self, _messages):
                llm.calls.append(schema.__name__)
                if schema not in llm._structured:          # e.g. rerank: not recorded -> code falls back
                    raise RuntimeError(f"no recorded reply for {schema.__name__}")
                return llm._structured[schema].model_copy(deep=True)

        return _Structured()

    def invoke(self, _messages):
        self.calls.append("text")
        return AIMessage(content=self._text)


@pytest.fixture()
def services(settings, fake_embeddings):
    products = discover_products(settings.data_dir)
    for p in products:
        ingest_product(p, settings, embeddings=fake_embeddings, build_graph=False)
    build_router_index(products, settings, fake_embeddings)
    svc = build_services(settings)
    svc.registry.embeddings = fake_embeddings
    svc._router = load_router(settings, fake_embeddings)
    return svc


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_brief_case(case, services, monkeypatch):
    recorded = RecordedLLM(case["recorded"])
    monkeypatch.setattr(llm_module, "get_chat_llm", lambda *a, **k: recorded)

    ctx = create_ticket(services, ComplaintRequest(customer_name=case["customer_name"],
                                                   order_id=case["order_id"] or None,
                                                   complaint=case["complaint"]))
    if case.get("approve"):          # the support person clicks "Approve" in the admin console
        monkeypatch.setattr(services.settings, "approval_timeout_s", 20)   # wait for the decision
        def approve_later():
            for _ in range(100):
                if ctx.ticket_id in services.memory.pending_approvals():
                    services.memory.decide_approval(ctx.ticket_id, "approve", "tester", None, None)
                    return
                time.sleep(0.05)

        threading.Thread(target=approve_later, daemon=True).start()
    status = asyncio.run(process_ticket(services, ctx))
    t = services.memory.get_ticket(ctx.ticket_id)
    exp = case["expected"]

    assert "QueryPlan" in recorded.calls                       # the recorded plan was really used
    assert status.startswith(exp["status"]), (status, t.escalation_reasons)
    assert t.category == exp["category"]
    assert t.product == _slug(exp["product"])
    assert t.language == exp["language"]
    if exp["reason"]:
        assert any(exp["reason"] in r for r in t.escalation_reasons), t.escalation_reasons
    if exp["status"] == "Pending Confirmation":               # solved: grounded, confident, cited
        assert t.confidence >= services.settings.confidence_threshold
        assert t.kb_sources and not t.escalation_reasons
    if exp["status"] == "Escalated":                          # held for a human, never auto-sent
        assert ctx.ticket_id in services.memory.pending_approvals() or t.status.startswith("Escalated")
