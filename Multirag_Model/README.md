# DocuGraph — Hybrid RAG over PDFs (LangChain · FAISS · Neo4j · FastAPI · Streamlit)

Ask questions about your PDFs. Answers come **only** from the indexed documents, retrieved by
fusing semantic search (FAISS + OpenAI embeddings), keyword search (BM25), entity search and an
LLM-planned, read-only Neo4j knowledge-graph traversal. Every request is traced in LangSmith.

---

## 1. Setup

Requires Python 3.11+ (tested on 3.13), Windows or Linux.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt

# Only needed for crawling JavaScript-rendered websites (optional):
python -m playwright install chromium

cp .env.example .env        # Windows: copy .env.example .env
# then edit .env: OPENAI_API_KEY, NEO4J_URI/USERNAME/PASSWORD, LANGSMITH_API_KEY
```

Neo4j is optional. Without it the app runs in `hybrid` mode (no graph). For Neo4j Aura use a
`neo4j+s://` URI — the connection is TLS-verified.

## 2. Run

| What | Command |
|---|---|
| Web UI (Streamlit) | `streamlit run streamlit_app.py` |
| Backend API (FastAPI) | `uvicorn hybrid_rag.api:app --reload` → docs at http://127.0.0.1:8000/docs |
| Ingest from the command line | `python -m hybrid_rag.ingest path/to/file.pdf --graph` (`--crawl` adds `SOURCE_WEBSITE`) |
| Sample PDF for a first try | `python scripts/make_sample_pdf.py` → `samples/sample_schemes.pdf` |
| Evaluation (CLI) | `python -m hybrid_rag.evaluation` |
| Offline tests (no API key) | `pytest` |
| Live LLM tests | set `OPENAI_API_KEY`, then `pytest tests/test_llm_live.py -v` |

**API quick reference**

```bash
curl -F "file=@samples/sample_schemes.pdf" -F "build_graph=true" http://127.0.0.1:8000/ingest
curl -N -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
     -d '{"question":"What subsidy does Green Farm Mission give?","mode":"hybrid+graph"}'
# /chat streams NDJSON events: session, plan, context, token..., final (or blocked)
# pass the returned session_id on the next call for conversation memory
curl -X POST "http://127.0.0.1:8000/evaluate?with_judge=true"
```

## 3. Architecture

```
hybrid_rag/
  config.py      settings (.env), prompts, TOKEN_PATTERN, SSL helpers, paths (pathlib)
  guardrails.py  input + output guardrails
  ingest.py      PDF text, table→record extraction, GPT-4o OCR (cached), KEY FACTS summaries,
                 chunking with ids/positions, BFS website crawler
  planner.py     QueryPlan (Pydantic) via ChatOpenAI.with_structured_output
  retrieval.py   RetrievalStore, 4 retrievers, RRF, neighbour expansion, context, streaming
                 answer, rag_events() pipeline
  graph.py       Neo4j build (write, ingestion only) + validated read-only traversal
  evaluation.py  hit-rate@k, MRR, LLM judge
  api.py         FastAPI backend
  ui.py          Streamlit UI (the only module that imports streamlit)
streamlit_app.py  entry point          fonts/  bundled Noto Sans Tamil (OFL)
eval/eval_set.json  test questions     samples/  generated sample PDF     tests/  pytest
```

Request flow (`retrieval.rag_events`, one LangSmith trace per request):

```
question ─► input guardrail ─► planner (structured output, uses chat memory)
        ─► graph_search: LLM plans Cypher ─► validate read-only ─► READ session
        ─► vector · BM25 · entity-summary · entity-name (· graph chunk ids)
        ─► Reciprocal Rank Fusion ─► top-k + neighbour chunks + full entity records + [G] facts
        ─► ChatOpenAI (streamed, no tools) ─► output guardrail ─► answer + evidence
```

The UI and the API are two front-ends over the same functions. The UI runs the pipeline
in-process and keeps the `RetrievalStore` in `st.session_state` (as required); the FastAPI
backend holds its own store and per-session memory for other clients (scripts, Postman, a JS
front-end). Both read/write the same persisted index under `data/index`.

## 4. How each requirement is implemented

**Modular, UI-free logic.** Ingestion, planning, retrieval, graph and evaluation never import
Streamlit. The store, LLM and Neo4j driver are function arguments, so `tests/` exercise the
pipeline with fake embeddings and a fake streaming chat model.

**Pinned dependencies.** `requirements.txt` lists every imported third-party package with exact
versions.

**Multilingual.** `config.TOKEN_PATTERN` (Latin, Tamil `\u0B80-\u0BFF`, Devanagari) is defined once
and used by BM25, entity matching, guardrails and language detection. The text splitter also
splits on the Indic danda. Tests cover Tamil BM25 retrieval and a Tamil eval question.

**Portability and bundled assets.** All paths use `pathlib` relative to the project. Noto Sans
Tamil ships in `fonts/` (with its OFL licence); the sample-PDF generator uses it to write Tamil
text, and the UI embeds it as the fallback web font so Tamil renders even without system fonts.

**Security.** No `verify=False` anywhere. `truststore` injects the OS certificate store; for sites
with broken chains set `CA_BUNDLE` to a bundled CA/intermediate `.pem`. Secrets only come from
`.env` (`.env` is git-ignored; `.env.example` documents the keys). Uploaded filenames are stripped
of client paths.

**Data ingestion.** `crawl_site` does a breadth-first crawl limited to the start domain, obeys
`robots.txt`, sleeps `CRAWL_DELAY` between requests, stops at `CRAWL_MAX_PAGES` and keeps a
`visited` set. `fetch_html` (requests + BeautifulSoup) is tried first; `fetch_with_playwright`
runs only when `needs_javascript` sees an empty/JS-only page. Tables (in HTML and in PDFs) become
one labelled `RECORD` per row (`rows_to_records`), with entity, category, source URL, linked
document URLs and date kept as separate metadata fields.

**Document processing.** `extract_pdf_pages` reads the text layer with PyMuPDF; pages with almost
no text are rendered and sent to GPT-4o (`ocr_page_with_vision`). OCR and summaries are cached in
`data/cache` keyed by SHA-256 of the image/text, so re-runs never pay twice. For non-English
documents `summarise_key_facts` writes an English KEY FACTS summary that is indexed with the
original. PDF table cells are rebuilt from word boxes (`table_cells_text`) because PyMuPDF's
table extractor can drop spaces.

**Query planning.** `plan_query` returns a validated `QueryPlan` (standalone question, entities,
category filter, needs_graph, keywords). A filter that is not a known category is discarded so a
hallucinated filter can't hide results; any planner failure falls back to the raw question.

**Context building.** `build_context` merges each hit with its previous/next chunks
(`neighbour_ids`, window configurable) in document order, adds the full record of every named
entity (`entity_records`), and appends graph facts as `[G]`. Blocks are labelled `[S1]…` for
citations.

**Hybrid retrieval.** `vector_search`, `bm25_search`, `entity_summary_search` and
`entity_name_search` are independent functions returning ranked chunk ids.
`reciprocal_rank_fusion` merges them (score = Σ 1/(k + rank), `RRF_K=60`) so FAISS distances and
BM25 scores never need normalising. `RETRIEVAL_MODE` (`vector`, `bm25`, `hybrid`, `hybrid+graph`)
is set in `.env` or the UI sidebar; in `hybrid+graph`, chunk ids returned by the graph join the
fusion as a fifth ranked list.

**Knowledge graph.** `build_graph` extracts entities/relations per chunk with structured output and
`MERGE`s `(:Chunk)-[:MENTIONS]->(:Entity)-[:REL {type, chunk_id}]->(:Entity)`. At query time
`plan_cypher` lets the LLM write the traversal, `validate_read_only_cypher` rejects anything with a
write/admin clause (CREATE, MERGE, SET, DELETE, CALL, LOAD CSV, apoc/dbms, multiple statements…),
ignores keywords inside string literals and enforces a LIMIT; the query then runs in a
`READ_ACCESS` session via `execute_read`, so Neo4j refuses writes too. If the planned query is
unsafe, fails or is empty, a parameterised 1–2-hop entity traversal is used instead.

**Guardrails.** Input: empty/too long, prompt injection, "search the web" requests, Cypher write
attempts, unsafe topics. Output: blocks prompt leakage, checks grounding (share of answer words
present in the retrieved context) and replaces ungrounded answers with the fixed "not found"
reply, and redacts emails, phone, Aadhaar and card numbers.

**Documents only, no web.** The answer prompt restricts the model to the context, no tools are
bound to ChatOpenAI, and if retrieval returns nothing the LLM isn't called at all.

**Memory and streaming.** The last `HISTORY_TURNS` messages feed both the planner (follow-up
rewriting) and the answer prompt. Answers stream token-by-token (`st.write_stream` in the UI,
NDJSON over FastAPI).

**Evaluation.** `evaluate_retrieval` reports hit-rate@k and MRR per mode on `eval/eval_set.json`
(relevance = chunk contains an `expected` string, which survives re-indexing). `llm_judge` scores
faithfulness, relevance and correctness 1–5 with structured output. `tests/test_llm_live.py` turns
this into automated LLM tests (quality thresholds, follow-up memory, refusal of out-of-scope
questions). Edit the eval set to match your own PDF.

**Observability.** With `LANGSMITH_API_KEY` set, `rag_request` is the parent run and planning,
graph planning/search, each retriever, generation, OCR, summaries and evaluation appear as nested
runs (`@traceable`); ChatOpenAI calls are traced automatically.

## 5. Notes

- `data/` (index, cache, uploads) is created on first run and git-ignored. Delete `data/index` to
  start fresh.
- `load_store` uses `allow_dangerous_deserialization=True` because FAISS persists a pickle; only
  load indexes this app wrote itself.
- The sample PDF content is fictional test data.
