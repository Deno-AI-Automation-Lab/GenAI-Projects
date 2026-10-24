"""Streamlit UI only. All logic lives in the other modules; this file just
wires widgets to them and keeps the store in st.session_state.

Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import base64
import html

import streamlit as st

from hybrid_rag import config
from hybrid_rag.ingest import ingest_pdf
from hybrid_rag.retrieval import (build_store, load_store, make_embeddings, make_llm,
                                  rag_events, save_store)

st.set_page_config(page_title="DocuGraph · Hybrid RAG", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# Styling (bundled Noto Sans Tamil is embedded as the fallback font)
# ---------------------------------------------------------------------------

def font_face_css() -> str:
    if not config.TAMIL_FONT.exists():
        return ""
    b64 = base64.b64encode(config.TAMIL_FONT.read_bytes()).decode("ascii")
    return (f"@font-face{{font-family:'NotoTamilLocal';"
            f"src:url(data:font/ttf;base64,{b64}) format('truetype');}}")


CSS = """
<style>
%FONT%
:root{--brand:#4f46e5;--brand2:#0ea5e9;--ink:#0f172a;--muted:#64748b;--card:#ffffff;--line:#e2e8f0}
html,body,[class*="css"],.stMarkdown,.stChatMessage{font-family:Inter,"Segoe UI",system-ui,'NotoTamilLocal',sans-serif}
.block-container{padding-top:4.5rem;max-width:1200px}  /* clear Streamlit's fixed top bar */
.nav{display:flex;justify-content:space-between;align-items:center;padding:.7rem 1.1rem;border-radius:14px;
     background:var(--card);border:1px solid var(--line);margin-bottom:1rem}
.nav .brand{font-weight:800;font-size:1.15rem;color:var(--ink)} .nav .brand span{color:var(--brand)}
.nav .links span{margin-left:1.2rem;color:var(--muted);font-size:.9rem}
.hero{padding:2rem 2.2rem;border-radius:18px;color:#fff;margin-bottom:1.2rem;
      background:linear-gradient(120deg,var(--brand),var(--brand2))}
.hero h1{margin:0;font-size:2rem;color:#fff} .hero p{margin:.4rem 0 0;opacity:.92}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-bottom:1rem}
.stat{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:.8rem 1rem}
.stat b{display:block;font-size:1.35rem;color:var(--ink)} .stat small{color:var(--muted)}
.chunk{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--brand);
       border-radius:12px;padding:.8rem 1rem;margin-bottom:.7rem}
.chunk.graph{border-left-color:#10b981}
.chunk .head{font-size:.8rem;color:var(--muted);margin-bottom:.35rem}
.chunk pre{white-space:pre-wrap;font-family:inherit;font-size:.88rem;margin:0;color:var(--ink)}
.badge{display:inline-block;padding:.08rem .5rem;border-radius:999px;font-size:.72rem;margin-right:.3rem;
       background:#eef2ff;color:var(--brand)} .badge.g{background:#ecfdf5;color:#047857}
.footer{color:var(--muted);font-size:.8rem;text-align:center;margin-top:2rem}
@media (max-width:700px){.stats{grid-template-columns:repeat(2,1fr)} .nav .links{display:none}}
@media (prefers-color-scheme:dark){:root{--ink:#e2e8f0;--card:#111827;--line:#1f2937;--muted:#94a3b8}}
</style>
"""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state() -> None:
    """Create models once per session; the retrieval store lives in session_state."""
    ss = st.session_state
    ss.setdefault("settings", config.get_settings())
    ss.setdefault("messages", [])
    ss.setdefault("last_events", {})
    if "llm" not in ss:
        ss.llm = make_llm(ss.settings)
        ss.embeddings = make_embeddings(ss.settings)
    if "store" not in ss:
        ss.store = load_store(ss.embeddings)   # reuse a previously persisted index
    if "driver" not in ss:
        ss.driver = None
        if ss.settings.graph_enabled:
            try:
                from hybrid_rag.graph import connect

                ss.driver = connect(ss.settings)
                ss.driver.verify_connectivity()
            except Exception as exc:
                ss.driver = None
                ss.graph_error = str(exc)


# ---------------------------------------------------------------------------
# Sidebar: upload, settings, status
# ---------------------------------------------------------------------------

def sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown("### 📄 Documents")
        files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
        do_graph = st.checkbox("Build knowledge graph (Neo4j)", value=ss.driver is not None,
                               disabled=ss.driver is None)
        if st.button("Index documents", type="primary", use_container_width=True, disabled=not files):
            index_files(files, do_graph)
        st.markdown("### ⚙️ Retrieval")
        modes = list(config.RETRIEVAL_MODES)
        ss.settings.retrieval_mode = st.selectbox("Mode", modes, index=modes.index(ss.settings.retrieval_mode))
        ss.settings.top_k = st.slider("Top-k chunks", 2, 10, ss.settings.top_k)
        ss.settings.neighbour_window = st.slider("Neighbour window", 0, 2, ss.settings.neighbour_window)
        if st.button("Clear conversation", use_container_width=True):
            ss.messages, ss.last_events = [], {}
            st.rerun()
        st.markdown("### 🔌 Status")
        st.caption(f"Neo4j: {'connected' if ss.driver else 'not configured / unreachable'}")
        if ss.get("graph_error"):
            st.caption(f"⚠️ {ss.graph_error[:120]}")
        st.caption(f"LangSmith tracing: {'on' if config.LANGSMITH_ENABLED else 'off'}")


def index_files(files, do_graph: bool) -> None:
    ss = st.session_state
    s = ss.settings
    vision = make_llm(s, model=s.vision_model)
    all_docs = []
    with st.status("Indexing…", expanded=True) as status:
        for f in files:
            path = config.UPLOAD_DIR / f.name
            path.write_bytes(f.getvalue())
            st.write(f"Parsing **{f.name}** (OCR for scanned pages)…")
            docs = ingest_pdf(path, s, ss.llm, vision)
            all_docs += docs
            st.write(f"→ {len(docs)} chunks / records")
        st.write("Embedding into FAISS + BM25…")
        ss.store = build_store(all_docs, ss.embeddings, existing=ss.store)
        save_store(ss.store)
        if do_graph and ss.driver is not None:
            from hybrid_rag.graph import build_graph

            st.write("Extracting entities & relations into Neo4j…")
            st.write(build_graph(ss.driver, all_docs, ss.llm, s.neo4j_database))
        status.update(label="Indexing complete", state="complete")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def header() -> None:
    ss = st.session_state
    st.markdown(CSS.replace("%FONT%", font_face_css()), unsafe_allow_html=True)
    st.markdown('<div class="nav"><div class="brand">Docu<span>Graph</span></div>'
                '<div class="links"><span>Chat</span><span>Evidence</span><span>Evaluation</span>'
                '<span>About</span></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>Ask your PDFs anything</h1><p>Hybrid retrieval — semantic '
                '(FAISS) + keyword (BM25) + knowledge graph (Neo4j) — fused with RRF. Answers come '
                'only from your documents.</p></div>', unsafe_allow_html=True)
    store = ss.store
    st.markdown(
        '<div class="stats">'
        f'<div class="stat"><b>{len(store.chunks)}</b><small>indexed chunks</small></div>'
        f'<div class="stat"><b>{len({d.metadata["doc_id"] for d in store.chunks})}</b><small>documents</small></div>'
        f'<div class="stat"><b>{ss.settings.retrieval_mode}</b><small>retrieval mode</small></div>'
        f'<div class="stat"><b>{"on" if ss.driver else "off"}</b><small>knowledge graph</small></div>'
        '</div>', unsafe_allow_html=True)


def chat_page() -> None:
    ss = st.session_state
    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    question = st.chat_input("Ask a question about your documents…")
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    events: dict = {}

    def token_stream():
        # Consume the pipeline's events; yield only tokens to st.write_stream.
        for ev in rag_events(question, ss.messages, ss.store, ss.llm, ss.settings, ss.driver):
            events[ev["type"]] = ev
            if ev["type"] == "token":
                yield ev["text"]
            elif ev["type"] == "blocked":
                yield f"⚠️ {ev['reason']}"

    with st.chat_message("assistant"):
        streamed = st.write_stream(token_stream())
        final = events.get("final")
        answer = final["answer"] if final else str(streamed)
        if final and final.get("replaced"):          # output guardrail rewrote the answer
            st.warning(f"Output guardrail: {final['guard']}")
            st.markdown(answer)
        elif final and not streamed:                 # no context -> fixed reply, no tokens
            st.markdown(answer)
        if "context" in events:
            st.caption(f"{len(events['context']['blocks'])} source blocks · "
                       f"{len(events['context']['graph']['rows'])} graph facts — see the Evidence tab")
    ss.messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    ss.last_events = events


def chunk_card(block: dict) -> str:
    m = block["meta"]
    badges = "".join(f'<span class="badge">{html.escape(k)} #{v}</span>' for k, v in block["ranks"].items())
    nb = f" · +{len(block['neighbours'])} neighbour(s)" if block["neighbours"] else ""
    return (f'<div class="chunk"><div class="head"><b>[{block["label"]}]</b> {html.escape(str(m.get("source")))} '
            f'· page {m.get("page")} · {m.get("kind")} · RRF {block["rrf"]}{nb}<br>{badges}</div>'
            f'<pre>{html.escape(block["text"][:1500])}</pre></div>')


def evidence_page() -> None:
    ev = st.session_state.last_events
    if "context" not in ev:
        st.info("Ask a question first — the retrieved chunks will appear here.")
        return
    if "plan" in ev:
        with st.expander("🧭 Query plan (structured output)", expanded=False):
            st.json(ev["plan"]["plan"])
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### 📚 Vector / keyword chunks")
        for b in ev["context"]["blocks"]:
            st.markdown(chunk_card(b), unsafe_allow_html=True)
    with right:
        st.markdown("#### 🕸️ Knowledge graph")
        g = ev["context"]["graph"]
        if g.get("cypher"):
            st.code(g["cypher"], language="cypher")
        if g.get("rows"):
            for r in g["rows"]:
                st.markdown(f'<div class="chunk graph"><span class="badge g">{html.escape(str(r.get("relation","")))}</span>'
                            f'<pre>{html.escape(str(r.get("source","")))} → {html.escape(str(r.get("target","")))}</pre>'
                            f'<div class="head">chunk {html.escape(str(r.get("chunk_id","")))}</div></div>',
                            unsafe_allow_html=True)
        else:
            st.caption("No graph facts for this question (graph off, or mode without graph).")
        if g.get("error"):
            st.caption(f"Planner note: {g['error']}")


def evaluation_page() -> None:
    from hybrid_rag.evaluation import evaluate_answers, evaluate_retrieval, load_eval_set

    ss = st.session_state
    st.markdown("#### Automated evaluation")
    st.caption(f"Eval set: `{config.EVAL_SET.relative_to(config.PROJECT_DIR)}` — edit it to match your PDF.")
    if ss.store.is_empty:
        st.info("Index a PDF first.")
        return
    data = load_eval_set()
    k = st.slider("k", 1, 10, 5)
    c1, c2 = st.columns(2)
    if c1.button("Run retrieval eval (hit-rate@k, MRR)", use_container_width=True):
        res = evaluate_retrieval(ss.store, data, ss.settings, k=k)
        st.dataframe([{"mode": m, **{x: y for x, y in v.items() if x != "ranks"}} for m, v in res.items()],
                     use_container_width=True)
    if c2.button("Run LLM judge", use_container_width=True):
        with st.spinner("Answering and grading each question…"):
            res = evaluate_answers(ss.store, data, ss.llm, ss.settings, ss.driver)
        st.json(res["summary"])
        st.dataframe(res["rows"], use_container_width=True)


def about_page() -> None:
    st.markdown("""
**Pipeline per question** — input guardrail → query planner (Pydantic structured output,
rewrites follow-ups using memory) → LLM-planned read-only Cypher traversal → vector + BM25 +
entity-summary + entity-name retrievers fused with Reciprocal Rank Fusion → context with
neighbouring chunks and full entity records → streamed ChatOpenAI answer → output guardrail.
Every step is traced in LangSmith. The model has no tools, so it cannot use the web.
""")


def main() -> None:
    try:
        init_state()
    except Exception as exc:
        st.error(f"Startup failed — check your .env (OPENAI_API_KEY): {exc}")
        st.stop()
    sidebar()
    header()
    tabs = st.tabs(["💬 Chat", "🔎 Evidence", "📊 Evaluation", "ℹ️ About"])
    with tabs[0]:
        chat_page()
    with tabs[1]:
        evidence_page()
    with tabs[2]:
        evaluation_page()
    with tabs[3]:
        about_page()
    st.markdown('<div class="footer">DocuGraph · LangChain · FAISS · Neo4j · LangSmith</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
