"""Live end-to-end test of the two custom test cases (needs OPENAI_API_KEY in .env, indexed
data and a running API).  Run:  python -m pytest tests/test_live_cases.py -v -s

A ticket the agents solve themselves ends in 'Pending Confirmation' (solution sent, waiting
for the customer's Yes); it becomes 'Resolved' only after the customer confirms.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from dotenv import dotenv_values

from support_agent.client import ApiError, SupportApiClient

# Read .env WITHOUT exporting it: exporting would leak real settings (API key, MAIL_ACCEPT_ALL,
# LangSmith) into the offline tests that run in the same pytest session.
_ENV = dotenv_values(Path(__file__).resolve().parent.parent / ".env")

CASES = json.loads((Path(__file__).parent / "test_cases.json").read_text(encoding="utf-8"))
API = os.getenv("API_BASE_URL") or _ENV.get("API_BASE_URL") or "http://127.0.0.1:8000"


def _api_up() -> bool:
    try:
        SupportApiClient(API, timeout=3).health()
        return True
    except ApiError:
        return False


pytestmark = [pytest.mark.live,
              pytest.mark.skipif(not (os.getenv("OPENAI_API_KEY") or _ENV.get("OPENAI_API_KEY")) or not _api_up(),
                                 reason="needs OPENAI_API_KEY and a running API")]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case(case):
    client = SupportApiClient(API)
    t = client.submit({k: case[k] for k in ("customer_name", "order_id", "complaint")} | {"channel": "api"})
    for _ in range(90):
        t = client.ticket(t["ticket_id"], case["order_id"])
        if t["status"] not in ("Processing",):
            break
        time.sleep(2)
    time.sleep(2)  # let the final writes of the Log Agent land before checking
    t = client.ticket(t["ticket_id"], case["order_id"])
    print(json.dumps(t, ensure_ascii=False, indent=1))
    if case["expected_status"] == "Resolved":
        # solved by the agents = solution sent and awaiting (or given) customer confirmation
        assert t["status"] in ("Pending Confirmation", "Resolved"), t["status"]
    else:
        assert t["status"].startswith(case["expected_status"]), t["status"]
    assert t["language"] == case["expected_language"]
    if case["expected_status"] == "Resolved":
        assert t["confidence"] >= 0.7 and t["kb_sources"]
    else:
        assert t["escalation_reasons"]
