"""Retrievers, rank fusion, context building and answer generation.

Nothing here touches Streamlit: the store, LLM and graph driver are passed in
as arguments, so every function runs the same from the UI, FastAPI, scripts
and pytest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import traceable
from rank_bm25 import BM25Okapi

from hybrid_rag import config
from hybrid_rag.config import Settings, tokenize
from hybrid_rag.guardrails import check_input, check_output
from hybrid_rag.planner import QueryPlan, plan_query

ENTITY_KINDS = ("record", "key_facts")

# ---------------------------------------------------------------------------
# Model factories (LangChain ChatOpenAI / OpenAIEmbeddings - no raw OpenAI SDK)
# ---------------------------------------------------------------------------

def make_llm(settings: Settings, model: str | None = None, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model or settings.chat_model, temperature=temperature, streaming=True)


def make_embeddings(settings: Settings):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embedding_model)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

@dataclass
class RetrievalStore:
    """All indexes for one corpus. Held by the caller (e.g. st.session_state)."""
    chunks: list[Document] = field(default_factory=list)
    vector: Any = None            # FAISS over all chunks
    entity_index: Any = None      # FAISS over records / KEY FACTS summaries
    bm25: Any = None              # BM25Okapi over all chunks (shared tokenizer)
    by_id: dict[str, Document] = field(default_factory=dict)
    by_pos: dict[tuple[str, int], str] = field(default_factory=dict)

    def reindex_lookups(self) -> None:
        """Rebuild id/position maps and the BM25 index after chunks change."""
        self.by_id = {d.metadata["chunk_id"]: d for d in self.chunks}
        self.by_pos = {(d.metadata["doc_id"], d.metadata["position"]): d.metadata["chunk_id"]
                       for d in self.chunks}
        corpus = [tokenize(d.page_content) or ["_"] for d in self.chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def categories(self) -> list[str]:
        return sorted({d.metadata.get("category", "") for d in self.chunks} - {""})

    def entities(self) -> list[str]:
        return sorted({d.metadata.get("entity", "") for d in self.chunks} - {""})


def build_store(docs: list[Document], embeddings, existing: RetrievalStore | None = None) -> RetrievalStore:
    """Add documents to (a copy of) a store: FAISS vector index, entity index,
    BM25 and lookup maps. Chunks already present (same chunk_id) are skipped."""
    store = existing or RetrievalStore()
    known = {d.metadata["chunk_id"] for d in store.chunks}
    new = [d for d in docs if d.metadata["chunk_id"] not in known]
    if new:
        ids = [d.metadata["chunk_id"] for d in new]
        if store.vector is None:
            store.vector = FAISS.from_documents(new, embeddings, ids=ids)
        else:
            store.vector.add_documents(new, ids=ids)
        ent = [d for d in new if d.metadata.get("kind") in ENTITY_KINDS]
        if ent:
            eids = [d.metadata["chunk_id"] for d in ent]
            if store.entity_index is None:
                store.entity_index = FAISS.from_documents(ent, embeddings, ids=eids)
            else:
                store.entity_index.add_documents(ent, ids=eids)
        store.chunks += new
    store.reindex_lookups()
    return store


def save_store(store: RetrievalStore, index_dir: Path = config.INDEX_DIR) -> None:
    """Persist FAISS indexes and chunk JSON under data/index."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    if store.vector is not None:
        store.vector.save_local(str(index_dir / "vector"))
    if store.entity_index is not None:
        store.entity_index.save_local(str(index_dir / "entity"))
    payload = [{"text": d.page_content, "meta": d.metadata} for d in store.chunks]
    (index_dir / "chunks.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_store(embeddings, index_dir: Path = config.INDEX_DIR) -> RetrievalStore:
    """Load a persisted store (empty store if none). The FAISS pickle is our own
    file written by save_store, hence allow_dangerous_deserialization."""
    index_dir = Path(index_dir)
    store = RetrievalStore()
    if not (index_dir / "chunks.json").exists():
        return store
    data = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    store.chunks = [Document(page_content=x["text"], metadata=x["meta"]) for x in data]
    if (index_dir / "vector").exists():
        store.vector = FAISS.load_local(str(index_dir / "vector"), embeddings,
                                        allow_dangerous_deserialization=True)
    if (index_dir / "entity").exists():
        store.entity_index = FAISS.load_local(str(index_dir / "entity"), embeddings,
                                              allow_dangerous_deserialization=True)
    store.reindex_lookups()
    return store


# ---------------------------------------------------------------------------
# Individual retrievers - each returns [(chunk_id, score)] best first
# ---------------------------------------------------------------------------

def _category_ok(doc: Document, category: str | None) -> bool:
    """Filter matches the category, or the chunk has no category (plain text)."""
    if not category:
        return True
    cat = doc.metadata.get("category", "")
    return not cat or cat.lower() == category.lower()


@traceable(name="vector_search", run_type="retriever")
def vector_search(store: RetrievalStore, query: str, k: int = 10,
                  category: str | None = None) -> list[tuple[str, float]]:
    """Semantic similarity over OpenAI embeddings (FAISS L2: lower = closer)."""
    if store.vector is None:
        return []
    hits = store.vector.similarity_search_with_score(query, k=k * 3)
    out = [(d.metadata["chunk_id"], float(s)) for d, s in hits if _category_ok(d, category)]
    return out[:k]


@traceable(name="bm25_search", run_type="retriever")
def bm25_search(store: RetrievalStore, query: str, k: int = 10,
                category: str | None = None) -> list[tuple[str, float]]:
    """Keyword search (BM25 Okapi) using the shared multilingual tokenizer."""
    if store.bm25 is None:
        return []
    q = tokenize(query)
    if not q:
        return []
    scores = store.bm25.get_scores(q)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order:
        if scores[i] <= 0:
            break
        d = store.chunks[i]
        if _category_ok(d, category):
            out.append((d.metadata["chunk_id"], float(scores[i])))
        if len(out) >= k:
            break
    return out


@traceable(name="entity_summary_search", run_type="retriever")
def entity_summary_search(store: RetrievalStore, query: str, k: int = 10,
                          category: str | None = None) -> list[tuple[str, float]]:
    """Semantic search restricted to entity records and KEY FACTS summaries."""
    if store.entity_index is None:
        return []
    hits = store.entity_index.similarity_search_with_score(query, k=k * 3)
    return [(d.metadata["chunk_id"], float(s)) for d, s in hits if _category_ok(d, category)][:k]


@traceable(name="entity_name_search", run_type="retriever")
def entity_name_search(store: RetrievalStore, query: str, entities: list[str] | None = None,
                       k: int = 10) -> list[tuple[str, float]]:
    """Exact entity-name match: a record whose full entity-name token sequence
    occurs in the question (or in a planner entity) scores 1.0."""
    q_tokens = " " + " ".join(tokenize(query)) + " "
    planned = [" " + " ".join(tokenize(e)) + " " for e in (entities or [])]
    out = []
    for d in store.chunks:
        name = d.metadata.get("entity", "")
        if not name or d.metadata.get("kind") not in ENTITY_KINDS:
            continue
        n = " " + " ".join(tokenize(name)) + " "
        if len(n.strip()) < 3:
            continue
        if n in q_tokens or any(n in p or (p.strip() and p in n) for p in planned):
            out.append((d.metadata["chunk_id"], 1.0))
    return out[:k]


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(ranked: dict[str, list[str]], k: int = 60) -> list[dict]:
    """RRF: score(d) = sum over lists of 1 / (k + rank). Rank-based, so BM25
    scores and FAISS distances never need to be put on the same scale."""
    fused: dict[str, dict] = {}
    for name, ids in ranked.items():
        for rank, cid in enumerate(dict.fromkeys(ids), start=1):
            item = fused.setdefault(cid, {"chunk_id": cid, "rrf": 0.0, "ranks": {}})
            item["rrf"] += 1.0 / (k + rank)
            item["ranks"][name] = rank
    return sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)


@traceable(name="retrieve", run_type="retriever")
def retrieve(store: RetrievalStore, plan: QueryPlan, settings: Settings, mode: str | None = None,
             graph_chunk_ids: list[str] | None = None) -> list[dict]:
    """Run the retrievers for the selected mode and fuse with RRF.
    Modes: vector | bm25 | hybrid | hybrid+graph (graph-linked chunks join the fusion)."""
    mode = mode or settings.retrieval_mode
    q, k, cat = plan.standalone_question, settings.per_retriever_k, plan.category_filter
    search_text = q if not plan.keywords else f"{q} {' '.join(plan.keywords)}"
    ranked: dict[str, list[str]] = {}
    if mode in ("vector", "hybrid", "hybrid+graph"):
        ranked["vector"] = [c for c, _ in vector_search(store, q, k, cat)]
    if mode in ("bm25", "hybrid", "hybrid+graph"):
        ranked["bm25"] = [c for c, _ in bm25_search(store, search_text, k, cat)]
    if mode in ("hybrid", "hybrid+graph"):
        ranked["entity_summary"] = [c for c, _ in entity_summary_search(store, q, k, cat)]
        ranked["entity_name"] = [c for c, _ in entity_name_search(store, q, plan.entities, k)]
    if mode == "hybrid+graph" and graph_chunk_ids:
        ranked["graph"] = [c for c in graph_chunk_ids if c in store.by_id]
    fused = reciprocal_rank_fusion({n: ids for n, ids in ranked.items() if ids}, settings.rrf_k)
    return fused[: settings.top_k]


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def neighbour_ids(store: RetrievalStore, chunk_id: str, window: int = 1) -> list[str]:
    """Ids of the chunk and its previous/next `window` chunks in the same document."""
    d = store.by_id.get(chunk_id)
    if d is None:
        return []
    doc_id, pos = d.metadata["doc_id"], d.metadata["position"]
    ids = [store.by_pos.get((doc_id, p)) for p in range(pos - window, pos + window + 1)]
    return [i for i in ids if i]


def entity_records(store: RetrievalStore, entities: list[str]) -> list[str]:
    """Chunk ids of the full record(s) for each named entity."""
    wanted = [" ".join(tokenize(e)) for e in entities if e]
    out = []
    for d in store.chunks:
        name = " ".join(tokenize(d.metadata.get("entity", "")))
        if name and d.metadata.get("kind") in ENTITY_KINDS and any(w and (w in name or name in w) for w in wanted):
            out.append(d.metadata["chunk_id"])
    return out


def build_context(store: RetrievalStore, hits: list[dict], plan: QueryPlan,
                  graph_rows: list[dict] | None, settings: Settings) -> tuple[str, list[dict]]:
    """Assemble labelled context blocks:
    [S#] = retrieved chunk merged with its neighbours (ordered by position);
    full entity records for named entities; [G] = graph traversal facts.
    Returns (context text, source blocks for display/citations)."""
    from hybrid_rag.graph import graph_rows_to_text

    blocks: list[dict] = []
    used: set[str] = set()
    for h in hits:
        ids = [i for i in neighbour_ids(store, h["chunk_id"], settings.neighbour_window) if i not in used]
        if h["chunk_id"] not in ids:
            continue
        used.update(ids)
        docs = sorted((store.by_id[i] for i in ids), key=lambda d: d.metadata["position"])
        blocks.append({"chunk_id": h["chunk_id"], "neighbours": [i for i in ids if i != h["chunk_id"]],
                       "text": "\n".join(d.page_content for d in docs),
                       "meta": store.by_id[h["chunk_id"]].metadata, "ranks": h.get("ranks", {}),
                       "rrf": round(h.get("rrf", 0.0), 5)})
    for cid in entity_records(store, plan.entities):
        if cid not in used:
            used.add(cid)
            d = store.by_id[cid]
            blocks.append({"chunk_id": cid, "neighbours": [], "text": d.page_content,
                           "meta": d.metadata, "ranks": {"entity_record": 1}, "rrf": 0.0})
    parts = []
    for n, b in enumerate(blocks, start=1):
        b["label"] = f"S{n}"
        m = b["meta"]
        parts.append(f"[S{n}] source={m.get('source')} page={m.get('page')} kind={m.get('kind')}\n{b['text']}")
    if graph_rows:
        parts.append("[G] Knowledge graph facts:\n" + graph_rows_to_text(graph_rows))
    return "\n\n".join(parts), blocks


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def history_messages(history: list[dict], turns: int) -> list:
    """Conversation memory: last `turns` messages as LangChain messages."""
    msgs = []
    for m in history[-turns:] if turns else []:
        msgs.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))
    return msgs


@traceable(name="generate_answer", run_type="llm")
def stream_answer(question: str, context: str, history: list[dict], llm,
                  settings: Settings) -> Iterator[str]:
    """Stream tokens from ChatOpenAI. The system prompt restricts the model to
    the retrieved CONTEXT; no tools are bound, so it cannot reach the web."""
    msgs = [SystemMessage(content=config.ANSWER_SYSTEM_PROMPT.format(context=context))]
    msgs += history_messages(history, settings.history_turns)
    msgs.append(HumanMessage(content=question))
    for chunk in llm.stream(msgs):
        if chunk.content:
            yield chunk.content


@traceable(name="rag_request", run_type="chain")
def rag_events(question: str, history: list[dict], store: RetrievalStore, llm, settings: Settings,
               graph_driver=None, mode: str | None = None) -> Iterator[dict]:
    """Full traced pipeline as an event stream (shared by UI, API and eval):
    input guard -> plan -> graph traversal -> hybrid retrieval + RRF ->
    context -> streamed answer -> output guard."""
    mode = mode or settings.retrieval_mode
    guard = check_input(question, settings.max_question_chars)
    if not guard.allowed:
        yield {"type": "blocked", "reason": guard.reason}
        return
    if store is None or store.is_empty:
        yield {"type": "blocked", "reason": "No documents indexed yet - upload a PDF first."}
        return
    plan = plan_query(guard.text, history, llm, store.categories(), settings)
    yield {"type": "plan", "plan": plan.model_dump()}

    graph = {"cypher": "", "rows": [], "error": ""}
    if mode == "hybrid+graph" and graph_driver is not None:
        from hybrid_rag.graph import graph_search

        graph = graph_search(graph_driver, plan.standalone_question, plan.entities, llm, settings)
    graph_ids = [r.get("chunk_id") for r in graph["rows"] if r.get("chunk_id")]
    hits = retrieve(store, plan, settings, mode, graph_ids)
    context, blocks = build_context(store, hits, plan, graph["rows"], settings)
    yield {"type": "context", "blocks": blocks, "graph": graph, "mode": mode}

    if not blocks and not graph["rows"]:
        yield {"type": "final", "answer": config.NO_ANSWER, "guard": "no context", "context": ""}
        return
    answer = ""
    for tok in stream_answer(plan.standalone_question, context, history, llm, settings):
        answer += tok
        yield {"type": "token", "text": tok}
    out = check_output(answer, context)
    yield {"type": "final", "answer": out.text, "guard": out.reason or "ok",
           "replaced": out.text != answer, "context": context}


def answer_question(question: str, history: list[dict], store: RetrievalStore, llm,
                    settings: Settings, graph_driver=None, mode: str | None = None) -> dict:
    """Non-streaming convenience wrapper: collects rag_events into one dict."""
    result: dict[str, Any] = {"answer": "", "blocks": [], "graph": {}, "plan": {}, "context": ""}
    for ev in rag_events(question, history, store, llm, settings, graph_driver, mode):
        if ev["type"] == "blocked":
            result.update(answer=ev["reason"], blocked=True)
        elif ev["type"] == "plan":
            result["plan"] = ev["plan"]
        elif ev["type"] == "context":
            result.update(blocks=ev["blocks"], graph=ev["graph"])
        elif ev["type"] == "final":
            result.update(answer=ev["answer"], guard=ev["guard"], context=ev.get("context", ""))
    return result
