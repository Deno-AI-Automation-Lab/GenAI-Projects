"""Shared fixtures. Offline tests use deterministic fake embeddings - no API key needed."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hybrid_rag.config import get_settings  # noqa: E402
from hybrid_rag.ingest import ingest_pdf  # noqa: E402
from hybrid_rag.retrieval import build_store  # noqa: E402


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    from scripts.make_sample_pdf import make_pdf

    return make_pdf(tmp_path_factory.mktemp("pdf") / "sample.pdf")


@pytest.fixture(scope="session")
def settings():
    return get_settings(retrieval_mode="hybrid", top_k=5, chunk_size=300, chunk_overlap=50)


@pytest.fixture(scope="session")
def docs(sample_pdf, settings):
    return ingest_pdf(sample_pdf, settings)          # no LLM -> no OCR / summaries


@pytest.fixture(scope="session")
def store(docs):
    return build_store(docs, DeterministicFakeEmbedding(size=64))


class StubStructured:
    """Mimics llm.with_structured_output(Model).invoke(...) for planner tests."""

    def __init__(self, result):
        self.result = result

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _msgs):
        return self.result
