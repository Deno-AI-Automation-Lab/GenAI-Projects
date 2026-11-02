"""Shared offline fixtures: sample .docx products, fake embeddings, isolated settings."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from support_agent import config as _config  # noqa: E402
from support_agent import llm as _llm  # noqa: E402
from support_agent.config import Settings  # noqa: E402

# Offline tests must never read the developer's real .env (API keys, MAIL_ACCEPT_ALL,
# LangSmith ...): otherwise they call OpenAI, slow down and test the wrong settings.
# The live tests (test_live_cases.py) read .env themselves and talk to the running API.
_config.Settings.model_config["env_file"] = None
_llm.load_dotenv = lambda *args, **kwargs: False
_config.get_settings.cache_clear()


def _make_doc(path: Path, title: str, blocks: list) -> None:
    """blocks: ("h", text) heading | ("p", text) paragraph | ("t", [[header], [row], ...]) table."""
    d = Document()
    d.add_heading(title, level=1)
    for kind, val in blocks:
        if kind == "h":
            d.add_heading(val, level=2)
        elif kind == "p":
            d.add_paragraph(val)
        else:
            t = d.add_table(rows=len(val), cols=len(val[0]))
            for r, row in enumerate(val):
                for c, cell in enumerate(row):
                    t.cell(r, c).text = cell
    path.parent.mkdir(parents=True, exist_ok=True)
    d.save(str(path))


@pytest.fixture(scope="session")
def sample_data(tmp_path_factory) -> Path:
    """Two small fake products so isolation and retrieval can be tested offline."""
    root = tmp_path_factory.mktemp("data")
    fryer = root / "SmartCrisp Pro 8L Wi-Fi Air Fryer"
    _make_doc(fryer / "shipping_policy.docx", "Shipping Policy", [
        ("h", "Delivery timelines"), ("p", "Orders ship within 2 business days. Metro delivery takes 3-5 days."),
        ("t", [["Region", "Delivery time", "Fee"], ["Metro", "3-5 days", "Free"], ["Rural", "7-10 days", "Rs 99"]]),
        ("h", "Delayed shipments"), ("p", "If your order is delayed beyond 10 days, contact support with your order ID "
                                          "and we will open a courier investigation within 24 hours."),
    ])
    _make_doc(fryer / "product_faq.docx", "Product FAQ", [
        ("p", "Why does my air fryer show error E3?"),
        ("p", "Error E3 means the basket is not inserted fully. Push the basket until it clicks."),
        ("p", "How do I connect the fryer to Wi-Fi?"),
        ("p", "Open the SmartCrisp app, hold the Wi-Fi button for 5 seconds and follow the pairing steps."),
    ])
    _make_doc(fryer / "refund_policy.docx", "Refund Policy", [
        ("h", "Refund window"), ("p", "Refunds are accepted within 30 days of delivery for unused products."),
        ("h", "Payment disputes"), ("p", "Duplicate charges and chargeback disputes are investigated by the payments team."),
    ])
    _make_doc(fryer / "account_and_login.docx", "Account and Login", [
        ("h", "Password reset"), ("p", "Use Forgot Password on the login page; the OTP is valid for 10 minutes."),
    ])
    phones = root / "AeroBeat X9 Wireless Noise-Cancelling Headphones"
    _make_doc(phones / "product_faq.docx", "Product FAQ", [
        ("p", "How do I reset the headphones?"),
        ("p", "Hold the power and ANC buttons together for 10 seconds until the LED flashes white."),
    ])
    _make_doc(phones / "shipping_policy.docx", "Shipping Policy", [
        ("h", "Delivery timelines"), ("p", "Headphones ship within 1 business day by express courier."),
    ])
    return root


@pytest.fixture()
def settings(tmp_path, sample_data) -> Settings:
    s = Settings(_env_file=None, OPENAI_API_KEY="", DATA_DIR=str(sample_data), INDEX_DIR=str(tmp_path / "idx"),
                 CACHE_DIR=str(tmp_path / "cache"), MEMORY_DB_PATH=str(tmp_path / "mem.db"),
                 TICKET_LOG_CSV=str(tmp_path / "log.csv"), GRAPH_ENABLED=False, RERANK_ENABLED=False,
                 OCR_ENABLED=False, EMAIL_ENABLED=False, APPROVAL_TIMEOUT_SECONDS=1, RETRIEVAL_MODE="hybrid",
                 SUPPORT_MAILBOX="denomelvin24@gmail.com")   # test inbox address (no default in code)
    s.ensure_dirs()
    return s


@pytest.fixture()
def fake_embeddings():
    from langchain_core.embeddings import DeterministicFakeEmbedding

    return DeterministicFakeEmbedding(size=64)

