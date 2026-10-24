"""Evaluation: retrieval metrics (hit-rate@k, MRR) and an LLM judge.

Eval set format (eval/eval_set.json):
[{"question": "...", "expected": ["text that a relevant chunk contains", ...],
  "reference": "optional reference answer"}]
A retrieved chunk counts as relevant if it contains any `expected` string
(case-insensitive) - this survives re-ingestion, unlike hard-coded chunk ids.
"""
from __future__ import annotations

import json
from pathlib import Path

from langsmith import traceable
from pydantic import BaseModel, Field

from hybrid_rag import config
from hybrid_rag.planner import QueryPlan
from hybrid_rag.retrieval import RetrievalStore, answer_question, retrieve


def is_relevant(text: str, expected: list[str]) -> bool:
    low = text.lower()
    return any(e.lower() in low for e in expected if e)


def first_relevant_rank(ranked_texts: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first relevant result, or None."""
    for i, t in enumerate(ranked_texts, start=1):
        if is_relevant(t, expected):
            return i
    return None


def hit_rate_at_k(ranks: list[int | None], k: int) -> float:
    """Share of questions with a relevant result in the top k."""
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Average of 1/rank of the first relevant result (0 when none)."""
    return sum(1.0 / r for r in ranks if r) / len(ranks) if ranks else 0.0


def load_eval_set(path: Path = config.EVAL_SET) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@traceable(name="evaluate_retrieval", run_type="chain")
def evaluate_retrieval(store: RetrievalStore, eval_set: list[dict], settings: config.Settings,
                       modes: tuple[str, ...] = ("vector", "bm25", "hybrid"), k: int = 5) -> dict:
    """hit-rate@k and MRR per retrieval mode, so approaches can be compared.
    Uses the raw question (no LLM planner) to keep the metric deterministic."""
    results = {}
    for mode in modes:
        s = config.get_settings(retrieval_mode=mode, top_k=k)
        ranks = []
        for item in eval_set:
            plan = QueryPlan(standalone_question=item["question"])
            hits = retrieve(store, plan, s, mode)
            texts = [store.by_id[h["chunk_id"]].page_content for h in hits]
            ranks.append(first_relevant_rank(texts, item["expected"]))
        results[mode] = {f"hit_rate@{k}": round(hit_rate_at_k(ranks, k), 3),
                         "mrr": round(mean_reciprocal_rank(ranks), 3), "ranks": ranks}
    return results


class JudgeScore(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    reason: str = ""


@traceable(name="llm_judge", run_type="llm")
def llm_judge(question: str, answer: str, context: str, llm, reference: str = "") -> JudgeScore:
    """LLM-as-judge with structured output (1-5 per criterion)."""
    judge = llm.with_structured_output(JudgeScore)
    return judge.invoke(config.JUDGE_PROMPT.format(
        question=question, reference=reference or "(none)", context=context[:8000], answer=answer))


@traceable(name="evaluate_answers", run_type="chain")
def evaluate_answers(store: RetrievalStore, eval_set: list[dict], llm, settings: config.Settings,
                     graph_driver=None) -> dict:
    """Run the full pipeline per question and grade it with the LLM judge."""
    rows = []
    for item in eval_set:
        res = answer_question(item["question"], [], store, llm, settings, graph_driver)
        score = llm_judge(item["question"], res["answer"], res.get("context", ""), llm,
                          item.get("reference", ""))
        rows.append({"question": item["question"], "answer": res["answer"], **score.model_dump()})
    n = len(rows) or 1
    summary = {c: round(sum(r[c] for r in rows) / n, 2) for c in ("faithfulness", "relevance", "correctness")}
    return {"summary": summary, "rows": rows}


def main() -> None:  # pragma: no cover - CLI: python -m hybrid_rag.evaluation
    from hybrid_rag.retrieval import load_store, make_embeddings, make_llm

    s = config.get_settings()
    store = load_store(make_embeddings(s))
    if store.is_empty:
        raise SystemExit("Index is empty - run `python -m hybrid_rag.ingest your.pdf` first.")
    data = load_eval_set()
    print(json.dumps(evaluate_retrieval(store, data, s), indent=2))
    print(json.dumps(evaluate_answers(store, data, make_llm(s), s)["summary"], indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
