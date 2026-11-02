"""Retrieval: per-product retrievers, RAG-Fusion, RRF, re-ranking, context building, answering.

Pipeline (one product only – indexes are physically separate per product):

  QueryPlan.fusion_queries ──┬─ vector_search       (FAISS chunks)
                             ├─ bm25_search         (multilingual BM25)
                             └─ entity_summary_search (FAISS record summaries)
  plan.entities ─────────────── exact_entity_search (title / phrase match)
  question ──────────────────── graph_search        (Neo4j, read-only)
                    ↓ reciprocal_rank_fusion
                    ↓ rerank_passages  (LLM 0-10, fallback = RRF order)
                    ↓ build_context    (+ prev/next neighbours, + full matched records)
                    ↓ generate_answer  (grounded, cited [S1].., in customer's language)

Every retriever is callable on its own; RETRIEVAL_MODE selects the combination.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from rank_bm25 import BM25Plus

from .config import ANSWER_SYSTEM, RERANK_SYSTEM, RetrievalMode, Settings, tokenize
from .models import Chunk, Passage, QueryPlan, RerankScores, RetrievalResult

log = logging.getLogger(__name__)

Ranked = list[str]  # ordered chunk ids, best first


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class ProductIndex:
    """Everything needed to search ONE product."""

    slug: str
    name: str
    chunks: dict[str, Chunk]
    order: list[str]
    records: dict[str, dict]
    bm25: BM25Plus
    bm25_terms: list[set[str]] = field(default_factory=list)
    vector_store: Any | None = None      # FAISS over chunks
    summary_store: Any | None = None     # FAISS over record summaries
    errors: list[str] = field(default_factory=list)


def build_bm25(chunks: list[Chunk]) -> BM25Plus:
    """BM25+ over title+text with the shared multilingual TOKEN_PATTERN.
    BM25+ keeps IDF positive, so small per-product corpora still score common terms."""
    corpus = [tokenize(f"{c.section_title} {c.text}") or ["_"] for c in chunks]
    return BM25Plus(corpus)


def make_product_index(slug: str, name: str, chunks: list[Chunk], records: list[dict],
                       vector_store=None, summary_store=None) -> ProductIndex:
    """Assemble an index from in-memory parts (used by loader and by tests)."""
    return ProductIndex(
        slug=slug, name=name, chunks={c.chunk_id: c for c in chunks}, order=[c.chunk_id for c in chunks],
        records={r["record_id"]: r for r in records}, bm25=build_bm25(chunks),
        bm25_terms=[set(tokenize(f"{c.section_title} {c.text}")) for c in chunks],
        vector_store=vector_store, summary_store=summary_store,
    )


def load_product_index(settings: Settings, slug: str, embeddings=None) -> ProductIndex:
    """Load chunks/records JSON and both FAISS stores for one product."""
    from langchain_community.vectorstores import FAISS

    from .llm import get_embeddings

    d = settings.index_dir / slug
    if not (d / "chunks.json").exists():
        raise FileNotFoundError(f"No index for product '{slug}'. Run: python -m support_agent.ingest")
    chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text(encoding="utf-8"))]
    records = json.loads((d / "records.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "product.json").read_text(encoding="utf-8"))
    idx = make_product_index(slug, meta["name"], chunks, records)
    try:
        emb = embeddings or get_embeddings()
        # Our own ingest step wrote these pickles; never load FAISS files from untrusted sources.
        idx.vector_store = FAISS.load_local(str(d / "faiss_chunks"), emb, allow_dangerous_deserialization=True)
        idx.summary_store = FAISS.load_local(str(d / "faiss_summaries"), emb, allow_dangerous_deserialization=True)
    except Exception as exc:  # semantic search unavailable -> BM25/exact still work
        log.error("FAISS load failed for %s: %s", slug, exc)
        idx.errors.append(f"faiss: {exc}")
    return idx


class IndexRegistry:
    """Thread-safe lazy cache of ProductIndex objects (one per product)."""

    def __init__(self, settings: Settings, embeddings=None):
        self.settings, self.embeddings = settings, embeddings
        self._cache: dict[str, ProductIndex] = {}
        self._lock = Lock()

    def get(self, slug: str) -> ProductIndex:
        with self._lock:
            if slug not in self._cache:
                self._cache[slug] = load_product_index(self.settings, slug, self.embeddings)
            return self._cache[slug]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# --------------------------------------------------------------------------- #
# Individual retrievers (each usable on its own)
# --------------------------------------------------------------------------- #
@traceable(name="vector_search", run_type="retriever")
def vector_search(idx: ProductIndex, query: str, k: int = 8, doc_type: str | None = None) -> Ranked:
    """Semantic similarity over this product's chunk FAISS index."""
    if idx.vector_store is None:
        return []
    kwargs = {"filter": {"doc_type": doc_type}, "fetch_k": k * 4} if doc_type else {}
    docs = idx.vector_store.similarity_search(query, k=k, **kwargs)
    return [d.metadata["chunk_id"] for d in docs]


@traceable(name="bm25_search", run_type="retriever")
def bm25_search(idx: ProductIndex, query: str, k: int = 8, doc_type: str | None = None) -> Ranked:
    """Keyword search (BM25Okapi) with the shared multilingual tokenizer."""
    q = tokenize(query)
    if not q:
        return []
    qset = set(q)
    scores = idx.bm25.get_scores(q)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in ranked:
        # BM25+ gives every doc a small floor score: require a real term overlap
        if not qset & idx.bm25_terms[i]:
            continue
        cid = idx.order[i]
        if doc_type and idx.chunks[cid].doc_type != doc_type:
            continue
        out.append(cid)
        if len(out) >= k:
            break
    return out


@traceable(name="entity_summary_search", run_type="retriever")
def entity_summary_search(idx: ProductIndex, query: str, k: int = 5) -> Ranked:
    """Search record (section) summaries, then expand each hit to its first chunks."""
    if idx.summary_store is None:
        return []
    out: Ranked = []
    for d in idx.summary_store.similarity_search(query, k=k):
        rec = idx.records.get(d.metadata["record_id"])
        if rec:
            out.extend(rec["chunk_ids"][:2])
    return list(dict.fromkeys(out))


@traceable(name="exact_entity_search", run_type="retriever")
def exact_entity_search(idx: ProductIndex, entities: list[str], k: int = 8) -> Ranked:
    """Exact entity-name matching against section titles and record text."""
    out: Ranked = []
    ents = [e.lower().strip() for e in entities if len(e.strip()) > 2]
    if not ents:
        return []
    scored = []
    for rec in idx.records.values():
        title, text = rec["title"].lower(), rec["text"].lower()
        score = sum(3 for e in ents if e in title) + sum(1 for e in ents if e in text)
        if score:
            scored.append((score, rec))
    for _, rec in sorted(scored, key=lambda x: x[0], reverse=True):
        out.extend(rec["chunk_ids"][:2])
        if len(out) >= k:
            break
    return out[:k]


def matched_record_ids(idx: ProductIndex, entities: list[str]) -> list[str]:
    """Records whose TITLE names an entity – their full text is added to context."""
    ents = [e.lower().strip() for e in entities if len(e.strip()) > 2]
    return [rid for rid, r in idx.records.items() if any(e in r["title"].lower() for e in ents)][:3]


@traceable(name="graph_search", run_type="retriever")
def graph_search(idx: ProductIndex, plan: QueryPlan, settings: Settings, driver=None) -> tuple[Ranked, list[str]]:
    """Neo4j traversal -> chunk ids of matched sections + relation facts."""
    from .graph import get_driver, graph_retrieve

    def _run(drv) -> tuple[Ranked, list[str]]:
        rows = graph_retrieve(drv, idx.slug, plan.standalone_question, plan.entities, settings)
        ids: Ranked = []
        facts: list[str] = []
        for r in rows:
            rec = idx.records.get(r["section_id"])
            if rec:
                ids.extend(rec["chunk_ids"][:2])
            facts.extend(f for f in r.get("facts", []) if f)
        return list(dict.fromkeys(ids)), list(dict.fromkeys(facts))[:12]

    if driver is not None:
        return _run(driver)
    with get_driver(settings) as drv:
        return _run(drv)


# --------------------------------------------------------------------------- #
# Fusion, re-ranking, context
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(ranked_lists: dict[str, list[Ranked]], k: int = 60) -> list[tuple[str, float, list[str]]]:
    """RRF: score(d) = Σ 1/(k + rank). Returns (chunk_id, score, retrievers) best first."""
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    for retriever, lists in ranked_lists.items():
        for ranked in lists:
            for rank, cid in enumerate(ranked):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
                sources.setdefault(cid, set()).add(retriever)
    return [(cid, s, sorted(sources[cid])) for cid, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


@traceable(name="rerank_passages")
def rerank_passages(question: str, passages: list[Passage], llm=None) -> list[Passage]:
    """LLM listwise scoring 0-10; on any failure keep the RRF order (fallback)."""
    if not passages:
        return passages
    from .llm import get_chat_llm

    try:
        scorer = (llm or get_chat_llm()).with_structured_output(RerankScores)
        listing = "\n\n".join(f"[{i}] ({p.source_file} – {p.section_title})\n{p.text[:700]}"
                              for i, p in enumerate(passages))
        res: RerankScores = scorer.invoke([
            SystemMessage(content=RERANK_SYSTEM),
            HumanMessage(content=f"Question: {question}\n\nPassages:\n{listing}\n\n"
                                 f"Return exactly {len(passages)} scores."),
        ])
        if len(res.scores) != len(passages):
            raise ValueError("score count mismatch")
        for p, s in zip(passages, res.scores):
            p.rerank_score = float(max(0, min(10, s)))
        return sorted(passages, key=lambda p: (p.rerank_score or 0, p.score), reverse=True)
    except Exception as exc:
        log.warning("Re-rank failed, using RRF order: %s", exc)
        return passages


def build_context(idx: ProductIndex, selected: list[Passage], matched_records: list[str],
                  max_chars: int = 9000) -> tuple[str, list[Passage]]:
    """Numbered context: each selected chunk with prev/next neighbours, plus full matched records."""
    blocks: list[str] = []
    used: set[str] = set()
    final: list[Passage] = []
    total = 0
    for p in selected:
        c = idx.chunks[p.chunk_id]
        parts = []
        for cid in (c.prev_id, c.chunk_id, c.next_id):
            if cid and cid in idx.chunks and cid not in used:
                nb = idx.chunks[cid]
                # Only stitch neighbours from the same source file (same document)
                if nb.source_file == c.source_file:
                    parts.append(nb.text)
                    used.add(cid)
        if not parts:
            continue
        text = "\n".join(parts)
        if total + len(text) > max_chars:
            break
        total += len(text)
        final.append(p)
        blocks.append(f"[S{len(final)}] (source: {c.source_file} | section: {c.section_title})\n{text}")
    for rid in matched_records:
        rec = idx.records.get(rid)
        if rec and not set(rec["chunk_ids"]) <= used and total + len(rec["text"]) <= max_chars:
            total += len(rec["text"])
            final.append(Passage(chunk_id=rec["chunk_ids"][0], source_file=rec["source_file"],
                                 section_title=rec["title"], text=rec["text"], retrievers=["full_record"]))
            blocks.append(f"[S{len(final)}] (source: {rec['source_file']} | full record: {rec['title']})\n{rec['text']}")
            used.update(rec["chunk_ids"])
    return "\n\n".join(blocks), final


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@traceable(name="hybrid_retrieve", run_type="chain")
def hybrid_retrieve(idx: ProductIndex, plan: QueryPlan, settings: Settings,
                    mode: RetrievalMode | None = None, driver=None, rerank_llm=None) -> RetrievalResult:
    """RAG-Fusion queries x retrievers -> RRF -> rerank -> context selection."""
    mode = RetrievalMode(mode or settings.retrieval_mode)
    k = settings.top_k_per_retriever
    queries = plan.fusion_queries or [plan.standalone_question]
    lists: dict[str, list[Ranked]] = {}
    errors: list[str] = list(idx.errors)
    facts: list[str] = []

    def run(name: str, fn, *args) -> None:
        try:
            res = fn(*args)
            if res:
                lists.setdefault(name, []).append(res)
        except Exception as exc:  # one failing retriever never breaks retrieval
            log.warning("%s failed: %s", name, exc)
            errors.append(f"{name}: {exc}")

    for q in queries:
        if mode in (RetrievalMode.VECTOR, RetrievalMode.HYBRID, RetrievalMode.HYBRID_GRAPH):
            run("vector", vector_search, idx, q, k)
            if plan.doc_type_filter:
                run("vector_filtered", vector_search, idx, q, k, plan.doc_type_filter)
        if mode in (RetrievalMode.BM25, RetrievalMode.HYBRID, RetrievalMode.HYBRID_GRAPH):
            run("bm25", bm25_search, idx, q, k)
        if mode in (RetrievalMode.HYBRID, RetrievalMode.HYBRID_GRAPH):
            run("entity_summary", entity_summary_search, idx, q, max(3, k // 2))
    if mode in (RetrievalMode.HYBRID, RetrievalMode.HYBRID_GRAPH):
        run("exact_entity", exact_entity_search, idx, plan.entities, k)
    if mode == RetrievalMode.HYBRID_GRAPH and settings.graph_enabled:
        try:
            ids, facts = graph_search(idx, plan, settings, driver)
            if ids:
                lists.setdefault("graph", []).append(ids)
        except Exception as exc:  # Neo4j down -> continue with hybrid only
            log.warning("graph retrieval unavailable: %s", exc)
            errors.append(f"graph: {exc}")
    # Last-resort fallback: if everything failed, BM25 on the raw question
    if not lists:
        run("bm25", bm25_search, idx, plan.standalone_question, k)

    fused = reciprocal_rank_fusion(lists, settings.rrf_k)[: k * 2]
    candidates = [
        Passage(chunk_id=cid, source_file=idx.chunks[cid].source_file,
                section_title=idx.chunks[cid].section_title, text=idx.chunks[cid].text,
                score=round(score, 5), retrievers=srcs)
        for cid, score, srcs in fused if cid in idx.chunks
    ]
    if settings.rerank_enabled:
        candidates = rerank_passages(plan.standalone_question, candidates[:12], rerank_llm)
        # Drop clearly irrelevant passages so "no answer" is detected honestly
        candidates = [p for p in candidates if p.rerank_score is None or p.rerank_score >= settings.min_rerank_score]
    selected = candidates[: settings.final_context_k]
    context, final = build_context(idx, selected, matched_record_ids(idx, plan.entities))
    if facts:
        context += "\n\n[Knowledge graph facts]\n" + "\n".join(f"- {f}" for f in facts)
    return RetrievalResult(product=idx.slug, query=plan.standalone_question, passages=final,
                           context=context, mode=mode.value, graph_facts=facts, errors=errors)


TONES: dict[str, str] = {
    "High": "urgent, sincerely apologetic and reassuring; acknowledge the frustration first and give clear next steps",
    "Medium": "warm, empathetic and solution-focused",
    "Low": "friendly, concise and helpful",
}


@traceable(name="generate_answer")
def generate_answer(question: str, retrieval: RetrievalResult, language: str, urgency: str,
                    company: str, llm=None) -> str:
    """Grounded answer from the selected context only (no outside knowledge)."""
    from .llm import get_chat_llm

    if not retrieval.passages:
        return ""
    llm = llm or get_chat_llm()
    msg = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM.format(company=company, language=language,
                                                   tone=TONES.get(urgency, TONES["Medium"]),
                                                   context=retrieval.context)),
        HumanMessage(content=question),
    ])
    return str(msg.content).strip()


def sources_of(retrieval: RetrievalResult) -> list[str]:
    """Unique 'file › section' labels for display and logging."""
    return list(dict.fromkeys(f"{p.source_file} › {p.section_title}" for p in retrieval.passages))
