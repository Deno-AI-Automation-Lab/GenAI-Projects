"""Evaluation: retrieval metrics (hit-rate@k, MRR) per mode + LLM-as-judge for answers.

Usage:
    python -m support_agent.evaluation make-golden --per-product 10   # synthetic set from your docs
    python -m support_agent.evaluation retrieval --k 5                # compares all retrieval modes
    python -m support_agent.evaluation answers --limit 10             # LLM judge on generated answers

Golden set format (eval/golden_set.jsonl), one JSON object per line:
    {"product": "<slug>", "question": "...", "relevant_record_ids": ["<record_id>", ...]}
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from statistics import mean

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from pydantic import BaseModel

from .config import JUDGE_SYSTEM, PROJECT_ROOT, RetrievalMode, Settings, get_settings
from .models import JudgeVerdict, QueryPlan

log = logging.getLogger(__name__)
GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_set.jsonl"


# ------------------------------- metrics --------------------------------- #
def hit_rate_at_k(ranked: list[list[str]], relevant: list[set[str]], k: int) -> float:
    """Share of queries with at least one relevant item in the top-k."""
    if not ranked:
        return 0.0
    return mean(1.0 if set(r[:k]) & rel else 0.0 for r, rel in zip(ranked, relevant))


def mean_reciprocal_rank(ranked: list[list[str]], relevant: list[set[str]], k: int | None = None) -> float:
    """Average of 1/rank of the first relevant item (0 if none in the top-k)."""
    if not ranked:
        return 0.0
    scores = []
    for r, rel in zip(ranked, relevant):
        rr = 0.0
        for i, item in enumerate(r[:k] if k else r, start=1):
            if item in rel:
                rr = 1.0 / i
                break
        scores.append(rr)
    return mean(scores)


def chunks_to_records(idx, chunk_ids: list[str]) -> list[str]:
    """Map ranked chunk ids to ranked unique record (section) ids."""
    return list(dict.fromkeys(idx.chunks[c].record_id for c in chunk_ids if c in idx.chunks))


# -------------------------------- judge ---------------------------------- #
@traceable(name="llm_judge")
def judge_answer(question: str, context: str, answer: str, llm=None) -> JudgeVerdict:
    """LLM-as-judge: faithfulness / relevance / completeness against the context."""
    from .llm import get_chat_llm

    judge = (llm or get_chat_llm(get_settings().judge_model)).with_structured_output(JudgeVerdict)
    return judge.invoke([
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"QUESTION:\n{question}\n\nCONTEXT:\n{context[:12000]}\n\nANSWER:\n{answer}"),
    ])


# ------------------------------ golden set ------------------------------- #
class _GenQ(BaseModel):
    question: str


def make_golden_set(settings: Settings, per_product: int = 10, seed: int = 7, out: Path = GOLDEN_PATH) -> int:
    """Create a synthetic golden set: sample sections and ask the LLM for a customer-style
    question each section answers. Review/edit the file before trusting the metrics."""
    from .llm import get_chat_llm

    llm = get_chat_llm(temperature=0.3).with_structured_output(_GenQ)
    rng = random.Random(seed)
    rows = []
    for d in sorted(p for p in settings.index_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        records = json.loads((d / "records.json").read_text(encoding="utf-8"))
        for rec in rng.sample(records, min(per_product, len(records))):
            q = llm.invoke([SystemMessage(content="Write one realistic customer support question that this "
                                                  "passage answers. Do not quote it verbatim."),
                            HumanMessage(content=f"{rec['title']}\n{rec['text'][:2000]}")])
            rows.append({"product": d.name, "question": q.question, "relevant_record_ids": [rec["record_id"]]})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return len(rows)


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------- runners --------------------------------- #
def evaluate_retrieval(golden: list[dict], registry, settings: Settings, k: int = 5,
                       modes: list[RetrievalMode] | None = None) -> dict[str, dict[str, float]]:
    """hit-rate@k and MRR@k per retrieval mode (rerank off for a fair comparison)."""
    from .retrieval import hybrid_retrieve

    modes = modes or list(RetrievalMode)
    s = settings.model_copy(update={"rerank_enabled": False, "final_context_k": k})
    report: dict[str, dict[str, float]] = {}
    for mode in modes:
        ranked, relevant = [], []
        for g in golden:
            idx = registry.get(g["product"])
            plan = QueryPlan(standalone_question=g["question"], product=g["product"], category="other",
                             urgency="Low", sentiment="Neutral", summary=g["question"],
                             entities=[], fusion_queries=[g["question"]])
            res = hybrid_retrieve(idx, plan, s, mode=mode)
            ranked.append(chunks_to_records(idx, [p.chunk_id for p in res.passages]))
            relevant.append(set(g["relevant_record_ids"]))
        report[mode.value] = {f"hit_rate@{k}": round(hit_rate_at_k(ranked, relevant, k), 3),
                              f"mrr@{k}": round(mean_reciprocal_rank(ranked, relevant, k), 3), "n": len(golden)}
    return report


def evaluate_answers(golden: list[dict], registry, settings: Settings, limit: int = 10) -> dict:
    """End-to-end answer quality with the LLM judge (faithfulness is the key metric)."""
    from .language import language_name
    from .models import QueryAnalysis
    from .resolution import draft_reply
    from .retrieval import hybrid_retrieve

    rows = []
    for g in golden[:limit]:
        idx = registry.get(g["product"])
        plan = QueryPlan(standalone_question=g["question"], product=g["product"], category="other",
                         urgency="Low", sentiment="Neutral", summary=g["question"], fusion_queries=[g["question"]])
        res = hybrid_retrieve(idx, plan, settings)
        analysis = QueryAnalysis(**plan.model_dump(), language="en", language_name=language_name("en"))
        draft = draft_reply(analysis, res, settings)
        verdict = judge_answer(g["question"], res.context, draft.body)
        rows.append(verdict.model_dump() | {"confidence": draft.confidence})
    return {m: round(mean(r[m] for r in rows), 3) for m in ("faithfulness", "relevance", "completeness", "confidence")} | {"n": len(rows)}


def main() -> None:
    from .llm import bootstrap_runtime
    from .retrieval import IndexRegistry

    bootstrap_runtime()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("make-golden")
    g.add_argument("--per-product", type=int, default=10)
    r = sub.add_parser("retrieval")
    r.add_argument("--k", type=int, default=5)
    a = sub.add_parser("answers")
    a.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()
    s = get_settings()
    if args.cmd == "make-golden":
        print(f"wrote {make_golden_set(s, args.per_product)} questions to {GOLDEN_PATH}")
        return
    reg = IndexRegistry(s)
    golden = load_golden()
    result = evaluate_retrieval(golden, reg, s, args.k) if args.cmd == "retrieval" else evaluate_answers(golden, reg, s, args.limit)
    print(json.dumps(result, indent=2))
    out = PROJECT_ROOT / "eval" / f"report_{args.cmd}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
