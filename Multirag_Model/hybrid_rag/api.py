"""FastAPI backend: same core functions as the UI, exposed over HTTP.

Run:  uvicorn hybrid_rag.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from hybrid_rag import config
from hybrid_rag.ingest import ingest_pdf
from hybrid_rag.retrieval import (RetrievalStore, answer_question, build_store, load_store,
                                  make_embeddings, make_llm, rag_events, save_store)


class AppState:
    """Process-wide state. Models are created lazily so the app imports
    (and tests run) without an API key."""

    def __init__(self) -> None:
        self.settings = config.get_settings()
        self._llm = self._emb = self._driver = None
        self.store: RetrievalStore | None = None
        self.sessions: dict[str, list[dict]] = {}   # conversation memory per session

    @property
    def llm(self):
        self._llm = self._llm or make_llm(self.settings)
        return self._llm

    @property
    def embeddings(self):
        self._emb = self._emb or make_embeddings(self.settings)
        return self._emb

    @property
    def driver(self):
        if self._driver is None and self.settings.graph_enabled:
            from hybrid_rag.graph import connect

            self._driver = connect(self.settings)
        return self._driver

    def get_store(self) -> RetrievalStore:
        if self.store is None:
            self.store = load_store(self.embeddings)
        return self.store


state = AppState()
app = FastAPI(title="Hybrid RAG API", version="1.0.0",
              description="PDF Q&A with FAISS + BM25 + Neo4j hybrid retrieval")


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    mode: Optional[str] = None


@app.get("/health")
def health() -> dict:
    store = state.store
    return {"status": "ok", "chunks": len(store.chunks) if store else None,
            "mode": state.settings.retrieval_mode, "graph": state.settings.graph_enabled,
            "langsmith": config.LANGSMITH_ENABLED}


@app.post("/ingest")
def ingest(file: UploadFile = File(...), build_graph: bool = Form(False)) -> dict:
    """Upload a PDF, index it in FAISS/BM25 and optionally the knowledge graph."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")
    dest = config.UPLOAD_DIR / Path(file.filename).name   # strip any client path
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    s = state.settings
    docs = ingest_pdf(dest, s, state.llm, make_llm(s, model=s.vision_model))
    state.store = build_store(docs, state.embeddings, existing=state.get_store())
    save_store(state.store)
    result = {"file": dest.name, "chunks_added": len(docs), "total_chunks": len(state.store.chunks)}
    if build_graph and state.driver is not None:
        from hybrid_rag.graph import build_graph as kg_build

        result["graph"] = kg_build(state.driver, docs, state.llm, s.neo4j_database)
    return result


def _validate_mode(mode: str | None) -> str | None:
    if mode and mode not in config.RETRIEVAL_MODES:
        raise HTTPException(400, f"mode must be one of {config.RETRIEVAL_MODES}")
    return mode


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """Streams NDJSON events: plan, context, token..., final (or blocked)."""
    mode = _validate_mode(req.mode)
    sid = req.session_id or str(uuid.uuid4())
    history = state.sessions.setdefault(sid, [])
    store = state.get_store()

    def gen():
        answer = ""
        yield json.dumps({"type": "session", "session_id": sid}) + "\n"
        for ev in rag_events(req.question, history, store, state.llm, state.settings, state.driver, mode):
            if ev["type"] == "final":
                answer = ev["answer"]
                ev = {k: v for k, v in ev.items() if k != "context"}
            elif ev["type"] == "blocked":
                answer = ev["reason"]
            yield json.dumps(ev, ensure_ascii=False, default=str) + "\n"
        history.extend([{"role": "user", "content": req.question}, {"role": "assistant", "content": answer}])

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/chat/sync")
def chat_sync(req: ChatRequest) -> dict:
    """Non-streaming variant (handy for scripts and tests)."""
    mode = _validate_mode(req.mode)
    sid = req.session_id or str(uuid.uuid4())
    history = state.sessions.setdefault(sid, [])
    res = answer_question(req.question, history, state.get_store(), state.llm, state.settings,
                          state.driver, mode)
    history.extend([{"role": "user", "content": req.question}, {"role": "assistant", "content": res["answer"]}])
    res.pop("context", None)
    return {"session_id": sid, **res}


@app.delete("/session/{session_id}")
def clear_session(session_id: str) -> dict:
    state.sessions.pop(session_id, None)
    return {"cleared": session_id}


@app.post("/evaluate")
def evaluate(with_judge: bool = False) -> dict:
    """Retrieval metrics per mode (+ LLM judge if requested) on eval/eval_set.json."""
    from hybrid_rag.evaluation import evaluate_answers, evaluate_retrieval, load_eval_set

    store = state.get_store()
    if store.is_empty:
        raise HTTPException(400, "Index is empty - ingest a PDF first")
    data = load_eval_set()
    out = {"retrieval": evaluate_retrieval(store, data, state.settings)}
    if with_judge:
        out["judge"] = evaluate_answers(store, data, state.llm, state.settings, state.driver)
    return out
