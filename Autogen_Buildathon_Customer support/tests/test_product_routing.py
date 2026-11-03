"""Product routing safety: never answer from a product the customer did not mention."""
from __future__ import annotations

import pytest

from support_agent.models import QueryPlan
from support_agent.planner import RouterIndex, analyze_complaint, product_evidence

PRODUCTS = [{"name": "AeroBeat X9 Wireless Noise-Cancelling Headphones", "slug": "aerobeat"},
            {"name": "PureSweep V12 Robot Vacuum & Mop", "slug": "puresweep"},
            {"name": "SmartCrisp Pro 8L Wi-Fi Air Fryer", "slug": "smartcrisp"}]


class FakePlanner:
    """Simulates an LLM that (wrongly) maps the complaint to a listed product."""

    def __init__(self, product: str, mentioned: str = ""):
        self.product, self.mentioned = product, mentioned

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _msgs):
        return QueryPlan(standalone_question="q", product=self.product, category="refund", urgency="Low",
                         sentiment="Neutral", summary="s", fusion_queries=["q"], mentioned_product=self.mentioned)


@pytest.mark.parametrize("text,planner_choice,expected", [
    ("I want to return my Samsung washing machine and get a refund", "puresweep", "unknown"),
    ("Refund for my iPhone 15 please", "aerobeat", "unknown"),
    ("please cancel my order and refund", "aerobeat", "unknown"),          # no product named at all
    ("My headset keeps disconnecting from my laptop", "aerobeat", "aerobeat"),
    ("the robot cleaner misses corners", "puresweep", "puresweep"),
    ("SmartCrisp shows E3", "unknown", "smartcrisp"),                       # explicit name always wins
])
def test_product_needs_evidence(settings, text, planner_choice, expected):
    router = RouterIndex(store=None, products=PRODUCTS)
    a = analyze_complaint(text, router, settings, llm=FakePlanner(planner_choice))
    assert a.product == expected


def test_unknown_product_gets_honest_reply_not_policy(settings):
    from support_agent.models import Passage, QueryAnalysis, RetrievalResult
    from support_agent.resolution import draft_reply, evaluate_escalation

    a = QueryAnalysis(standalone_question="q", product="unknown", category="refund", urgency="Low",
                      sentiment="Neutral", summary="s", fusion_queries=["q"], mentioned_product="Samsung washing machine")
    res = RetrievalResult(product="unknown", query="q", passages=[], context="", mode="none")
    d = draft_reply(a, res, settings, supported_products=[p["name"] for p in PRODUCTS])
    assert not d.found_relevant_info and "AeroBeat X9" in d.body and "Samsung washing machine" in d.body
    assert "30 calendar days" not in d.body                                  # no policy facts
    reasons = evaluate_escalation(a, res, d, settings).reasons
    assert any("not supported" in r and "Samsung" in r for r in reasons)


def test_approved_unsupported_reply_closes_without_confirmation(settings):
    from support_agent.agents import TicketContext, apply_decision
    from support_agent.models import DraftReply, QueryAnalysis
    from support_agent.services import build_services

    svc = build_services(settings)
    tid = svc.memory.create_ticket(customer_name="A", order_id=None, complaint="washing machine refund",
                                   customer_email=None, channel="web")
    svc.memory.update_ticket(tid, status="Escalated - Awaiting Approval")
    svc.memory.request_approval(tid)
    svc.memory.decide_approval(tid, "approve", "admin", None, None)
    a = QueryAnalysis(standalone_question="q", product="unknown", category="refund", urgency="Low",
                      sentiment="Neutral", summary="s", fusion_queries=["q"])
    ctx = TicketContext(services=svc, ticket_id=tid, customer_name="A", order_id=None, complaint="x",
                        customer_email=None, analysis=a, draft=DraftReply(subject="", body="We support only ...", confidence=0))
    out = apply_decision(ctx, "approve", None)
    t = svc.memory.get_ticket(tid)
    assert t.status == "Closed - Not Supported" and "Did this solve" not in (t.reply_body or "")
    assert "Closed" in out


def test_generic_words_never_decide_the_product():
    """'air' / 'robot' / 'noise' alone must not route to a product (exact match overrides the planner)."""
    from support_agent.planner import exact_product_mention

    products = [{"name": "AeroBeat X9 Wireless Noise-Cancelling Headphones", "slug": "aerobeat"},
                {"name": "PureSweep V12 Robot Vacuum & Mop", "slug": "puresweep"},
                {"name": "SmartCrisp Pro 8L Wi-Fi Air Fryer", "slug": "smartcrisp"}]
    assert exact_product_mention("I want a refund for my Philips air purifier", products) is None
    assert exact_product_mention("my Dyson robot lawn mower is noisy", products) is None
    assert exact_product_mention("noise in my Bose headphones", products) is None
    assert exact_product_mention("my SmartCrisp shows E3", products) == "smartcrisp"
    assert exact_product_mention("the V12 will not dock", products) == "puresweep"


def test_store_brand_routes_to_its_product(tmp_path):
    """'my SoundSphere account' -> AeroBeat: store names come from each product's own documents."""
    import json

    from support_agent.config import Settings
    from support_agent.planner import _add_store_aliases, exact_product_mention

    s = Settings(_env_file=None, INDEX_DIR=str(tmp_path))
    texts = {"aerobeat": "Store: SoundSphere Online", "puresweep": "HomePilot Online | Knowledge Base",
             "smartcrisp": "KitchenNova Online account guide"}
    products = [{"name": n, "slug": k} for k, n in (("aerobeat", "AeroBeat X9 Headphones"),
                                                   ("puresweep", "PureSweep V12 Robot Vacuum"),
                                                   ("smartcrisp", "SmartCrisp Pro 8L Air Fryer"))]
    for slug, text in texts.items():
        (tmp_path / slug).mkdir()
        (tmp_path / slug / "records.json").write_text(json.dumps([{"title": "x", "text": text}]))
    _add_store_aliases(s, products)
    assert exact_product_mention("my SoundSphere account is locked", products) == "aerobeat"
    assert exact_product_mention("KitchenNova app login fails", products) == "smartcrisp"
