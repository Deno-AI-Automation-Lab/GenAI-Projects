"""Ingestion + retrieval + full pipeline fallback, offline (fake embeddings, no LLM key)."""
from __future__ import annotations

import asyncio

import pytest

from support_agent.config import RetrievalMode, product_slug
from support_agent.ingest import (build_router_index, discover_products, doc_type_from_filename, ingest_product,
                                  parse_docx_sections)
from support_agent.models import ComplaintRequest, QueryPlan
from support_agent.retrieval import (bm25_search, exact_entity_search, hybrid_retrieve, load_product_index,
                                     vector_search)

FRYER = product_slug("SmartCrisp Pro 8L Wi-Fi Air Fryer")
PHONES = product_slug("AeroBeat X9 Wireless Noise-Cancelling Headphones")


@pytest.fixture()
def indexed(settings, fake_embeddings):
    products = discover_products(settings.data_dir)
    for p in products:
        ingest_product(p, settings, embeddings=fake_embeddings, build_graph=False)
    build_router_index(products, settings, fake_embeddings)
    return settings


def _plan(q: str, product: str, entities=None) -> QueryPlan:
    return QueryPlan(standalone_question=q, product=product, category="other", urgency="Low", sentiment="Neutral",
                     summary=q, entities=entities or [], fusion_queries=[q])


def test_discovery_and_doc_types(settings):
    names = [p.name for p in discover_products(settings.data_dir)]
    assert len(names) == 2
    from pathlib import Path
    assert doc_type_from_filename(Path("Refund Policy - X9.docx")) == "refund_policy"


def test_docx_sections_tables_and_faq(settings):
    folder = settings.data_dir / "SmartCrisp Pro 8L Wi-Fi Air Fryer"
    secs = parse_docx_sections(folder / "shipping_policy.docx")
    titles = [s.title for s in secs]
    assert "Delivery timelines" in titles and "Delayed shipments" in titles
    rows = [r for s in secs for r in s.table_rows]
    assert any("Region: Rural" in r and "Fee: Rs 99" in r for r in rows)  # one labelled record per row
    faq = parse_docx_sections(folder / "product_faq.docx")
    assert any(s.title.startswith("Why does my air fryer show error E3") for s in faq)


def test_chunks_have_neighbour_links(indexed, fake_embeddings):
    idx = load_product_index(indexed, FRYER, fake_embeddings)
    ship = [cid for cid in idx.order if ":shipping_policy:" in cid]
    first, second = idx.chunks[ship[0]], idx.chunks[ship[1]]
    assert first.prev_id is None and first.next_id == second.chunk_id and second.prev_id == first.chunk_id


def test_each_retriever_runs_alone_and_stays_in_product(indexed, fake_embeddings):
    idx = load_product_index(indexed, FRYER, fake_embeddings)
    assert bm25_search(idx, "error E3 basket")[0].startswith(FRYER)
    assert all(c.startswith(FRYER) for c in vector_search(idx, "delivery delayed"))
    assert exact_entity_search(idx, ["delayed shipments"])
    phones = load_product_index(indexed, PHONES, fake_embeddings)
    assert all(c.startswith(PHONES) for c in bm25_search(phones, "delivery courier"))


@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_hybrid_modes_and_context(indexed, fake_embeddings, mode):
    idx = load_product_index(indexed, FRYER, fake_embeddings)
    res = hybrid_retrieve(idx, _plan("my air fryer shows error E3", FRYER, ["error e3"]), indexed, mode=mode)
    assert res.passages, mode
    assert all(p.chunk_id.startswith(FRYER) for p in res.passages)
    if mode in (RetrievalMode.BM25, RetrievalMode.HYBRID):
        assert "E3" in res.context and "[S1]" in res.context


def test_bm25_finds_tamil(indexed, fake_embeddings):
    from support_agent.models import Chunk
    from support_agent.retrieval import make_product_index

    c = Chunk(chunk_id="t:0", product="t", product_name="T", doc_type="product_faq", source_file="f.docx",
              record_id="r", text="பணம் திரும்ப 30 நாட்களில் வழங்கப்படும்")
    idx = make_product_index("t", "T", [c], [{"record_id": "r", "title": "x", "text": c.text,
                                               "source_file": "f.docx", "doc_type": "product_faq", "chunk_ids": ["t:0"]}])
    assert bm25_search(idx, "பணம் திரும்ப") == ["t:0"]


def test_pipeline_fallback_without_llm_escalates_and_logs(indexed, fake_embeddings):
    """No OpenAI key: the team is skipped, fallback runs every step, ticket is escalated
    (no grounded answer possible) and written to the CSV log + memory."""
    from support_agent.pipeline import create_ticket, process_ticket
    from support_agent.services import build_services

    svc = build_services(indexed)
    svc.registry.embeddings = fake_embeddings
    from support_agent.planner import load_router
    svc._router = load_router(indexed, fake_embeddings)
    ctx = create_ticket(svc, ComplaintRequest(customer_name="Ravi", order_id="ORD-10001",
                                              complaint="My SmartCrisp air fryer shows error E3 every time"))
    status = asyncio.run(process_ticket(svc, ctx))
    assert status == "Escalated - Awaiting Approval"
    t = svc.memory.get_ticket(ctx.ticket_id)
    assert t.product == FRYER and t.escalation_reasons
    rows = svc.logger.read_csv()
    assert rows and rows[0]["ticket_id"] == ctx.ticket_id and rows[0]["status"].startswith("Escalated")
    agents = {e.agent for e in svc.memory.list_events(ctx.ticket_id)}
    assert {"Input_Guardrail", "Fallback", "Support_Approver"} <= agents
