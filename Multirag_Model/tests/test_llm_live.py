"""Automated LLM tests (live). Skipped unless OPENAI_API_KEY is set.
Runs the real pipeline on the sample PDF and grades it with the LLM judge.

    pytest tests/test_llm_live.py -v
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")


@pytest.fixture(scope="module")
def live(sample_pdf):
    from hybrid_rag.config import get_settings
    from hybrid_rag.ingest import ingest_pdf
    from hybrid_rag.retrieval import build_store, make_embeddings, make_llm

    s = get_settings(retrieval_mode="hybrid")
    llm = make_llm(s)
    store = build_store(ingest_pdf(sample_pdf, s, llm), make_embeddings(s))
    return s, llm, store


def test_hybrid_retrieval_quality(live):
    from hybrid_rag.evaluation import evaluate_retrieval, load_eval_set

    s, _, store = live
    res = evaluate_retrieval(store, load_eval_set(), s, modes=("hybrid",), k=5)
    assert res["hybrid"]["hit_rate@5"] >= 0.8
    assert res["hybrid"]["mrr"] >= 0.5


def test_answer_is_grounded_and_judged(live):
    from hybrid_rag.evaluation import llm_judge
    from hybrid_rag.retrieval import answer_question

    s, llm, store = live
    q = "What subsidy does Green Farm Mission give for drip irrigation?"
    res = answer_question(q, [], store, llm, s)
    assert "25,000" in res["answer"]
    score = llm_judge(q, res["answer"], res["context"], llm, "Rs. 25,000 per acre")
    assert score.faithfulness >= 4 and score.correctness >= 4


def test_follow_up_uses_memory(live):
    from hybrid_rag.retrieval import answer_question

    s, llm, store = live
    hist = [{"role": "user", "content": "Tell me about Solar Pump Support"},
            {"role": "assistant", "content": "It is a subsidy scheme for solar water pumps."}]
    res = answer_question("Which department issues it?", hist, store, llm, s)
    assert "Energy" in res["answer"]


def test_refuses_out_of_scope(live):
    from hybrid_rag.config import NO_ANSWER
    from hybrid_rag.retrieval import answer_question

    s, llm, store = live
    res = answer_question("Who won the 2022 FIFA World Cup?", [], store, llm, s)
    assert res["answer"] == NO_ANSWER
