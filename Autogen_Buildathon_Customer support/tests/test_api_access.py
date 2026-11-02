"""API access control: ticket log and dashboard are admin-only; a ticket is visible only
with its matching order ID (or an admin token)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    for k, v in {"MEMORY_DB_PATH": tmp_path / "m.db", "TICKET_LOG_CSV": tmp_path / "log.csv",
                 "INDEX_DIR": tmp_path / "idx", "CACHE_DIR": tmp_path / "cache", "OPENAI_API_KEY": "",
                 "EMAIL_ENABLED": "false", "GRAPH_ENABLED": "false", "APPROVAL_TIMEOUT_SECONDS": "1",
                 "ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": "Learning", "ADMIN_TOKEN_SECRET": "test-secret"}.items():
        monkeypatch.setenv(k, str(v))
    from support_agent.config import get_settings

    get_settings.cache_clear()
    from support_agent.api import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def _admin_headers(c: TestClient) -> dict:
    tok = c.post("/admin/login", json={"username": "admin", "password": "Learning"}).json()["token"]
    return {"Authorization": f"Bearer {tok}"}


def test_ticket_log_and_dashboard_are_admin_only(client):
    for path in ("/admin/log", "/admin/log.csv", "/admin/dashboard", "/admin/approvals"):
        assert client.get(path).status_code == 401, path
    assert client.get("/log").status_code == 404           # old public route is gone
    h = _admin_headers(client)
    assert client.get("/admin/log", headers=h).status_code == 200
    assert client.get("/admin/log.csv", headers=h).status_code == 200
    assert client.get("/admin/dashboard", headers=h).status_code == 200


def test_wrong_admin_password_rejected(client):
    assert client.post("/admin/login", json={"username": "admin", "password": "nope"}).status_code == 401
    assert client.get("/admin/log", headers={"Authorization": "Bearer forged"}).status_code == 401


def test_ticket_needs_matching_order_id(client):
    t = client.post("/tickets", json={"customer_name": "Ravi", "order_id": "ORD-104522",
                                      "complaint": "My air fryer shows error E3"}).json()
    tid = t["ticket_id"]
    assert client.get(f"/tickets/{tid}").status_code == 404
    assert client.get(f"/tickets/{tid}", params={"order_id": "ORD-999999"}).status_code == 404
    assert client.get(f"/tickets/{tid}", params={"order_id": "ord-104522"}).status_code == 200
    assert client.get(f"/tickets/{tid}/events", params={"order_id": "ORD-104522"}).status_code == 200
    assert client.get(f"/tickets/{tid}/events").status_code == 404
    assert client.get(f"/tickets/{tid}", headers=_admin_headers(client)).status_code == 200
    assert client.get("/tickets/TKT-20990101-000000", params={"order_id": "ORD-104522"}).status_code == 404


def test_resolve_endpoint_is_admin_only(client):
    assert client.get("/admin/escalations").status_code == 401
    assert client.post("/admin/tickets/TKT-1/resolve", json={"mode": "close"}).status_code == 401
    h = _admin_headers(client)
    assert client.get("/admin/escalations", headers=h).status_code == 200
    assert client.post("/admin/tickets/TKT-NOPE/resolve", json={"mode": "close"}, headers=h).status_code == 404


def test_ticket_without_order_opens_with_access_code(client):
    t = client.post("/tickets", json={"customer_name": "Ravi", "complaint": "PureSweep runs only one hour"}).json()
    assert t["order_id"] is None and t["access_code"]
    tid = t["ticket_id"]
    assert client.get(f"/tickets/{tid}", params={"code": t["access_code"].lower()}).status_code == 200
    assert client.get(f"/tickets/{tid}", params={"code": "WRONG1"}).status_code == 404


def test_unexpected_error_returns_json_500_with_error_id(client):
    """A bug inside an endpoint must not leak a traceback: JSON 500 + reference ID."""
    from support_agent import api

    class Broken:
        @property
        def router(self):
            raise RuntimeError("simulated bug")

    api.app.dependency_overrides[api.services] = lambda: Broken()
    try:
        r = TestClient(api.app, raise_server_exceptions=False).get("/products")
    finally:
        api.app.dependency_overrides.pop(api.services, None)
    assert r.status_code == 500
    body = r.json()
    assert body["error_id"] and "simulated bug" not in body["detail"]


def test_storage_error_returns_503(client, monkeypatch):
    from support_agent import api
    from support_agent.memory import MemoryStore, MemoryStoreError

    def locked(*_a, **_k):
        raise MemoryStoreError("Ticket storage is temporarily unavailable (database is locked)")

    monkeypatch.setattr(MemoryStore, "get_ticket", locked)
    r = TestClient(api.app, raise_server_exceptions=False).get("/tickets/TKT-1?code=ABC123")
    assert r.status_code == 503 and r.headers.get("retry-after") == "5"


def test_no_password_or_mailbox_is_hard_coded():
    """Secrets and personal addresses come only from .env: the code defaults are empty."""
    from support_agent.config import Settings

    s = Settings(_env_file=None)
    assert s.admin_password.get_secret_value() == ""
    assert s.openai_api_key.get_secret_value() == "" and s.support_mailbox_password.get_secret_value() == ""
    assert s.support_mailbox == "" and s.escalation_email == ""


def test_admin_login_disabled_without_password(client, monkeypatch):
    from support_agent.config import get_settings

    monkeypatch.setenv("ADMIN_PASSWORD", "")
    get_settings.cache_clear()
    r = client.post("/admin/login", json={"username": "admin", "password": "anything"})
    assert r.status_code == 503 and "ADMIN_PASSWORD" in r.json()["detail"]
