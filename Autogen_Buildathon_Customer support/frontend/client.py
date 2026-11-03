"""Thin HTTP client for the FastAPI backend (used by the Streamlit UI and scripts).

TLS verification stays ON (requests default); a custom CA can be supplied via
REQUESTS_CA_BUNDLE / CA_BUNDLE in .env.
"""
from __future__ import annotations

from typing import Any

import requests


class ApiError(RuntimeError):
    pass


class SupportApiClient:
    def __init__(self, base_url: str, token: str | None = None, timeout: int = 30):
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.session = requests.Session()

    def _req(self, method: str, path: str, **kw) -> Any:
        headers = kw.pop("headers", {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            r = self.session.request(method, f"{self.base}{path}", headers=headers, timeout=self.timeout, **kw)
        except requests.RequestException as exc:
            raise ApiError(f"Support service is not reachable at {self.base}. Start the API and retry.") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = r.text[:200]
            raise ApiError(f"{r.status_code}: {detail}")
        return r.content if path.endswith(".csv") else r.json()

    # public
    def health(self) -> dict: return self._req("GET", "/health")
    def submit(self, payload: dict) -> dict: return self._req("POST", "/tickets", json=payload)
    # customers send the ticket's access code or order ID (admins are authorised by token instead)
    def ticket(self, tid: str, order_id: str | None = None) -> dict:
        return self._req("GET", f"/tickets/{tid}", params={"code": order_id})
    def events(self, tid: str, after: int = 0, order_id: str | None = None) -> list[dict]:
        return self._req("GET", f"/tickets/{tid}/events", params={"after": after, "code": order_id})
    def chat(self, tid: str, message: str, order_id: str | None = None) -> dict:
        return self._req("POST", f"/tickets/{tid}/chat", params={"code": order_id},
                         json={"ticket_id": tid, "message": message})
    def chat_history(self, tid: str, order_id: str | None = None) -> list[dict]:
        return self._req("GET", f"/tickets/{tid}/chat", params={"code": order_id})

    def confirm(self, tid: str, resolved: bool, comment: str | None = None, order_id: str | None = None) -> dict:
        return self._req("POST", f"/tickets/{tid}/confirm", params={"code": order_id},
                         json={"resolved": resolved, "comment": comment})

    # admin
    def login(self, username: str, password: str) -> str:
        self.token = self._req("POST", "/admin/login", json={"username": username, "password": password})["token"]
        return self.token
    def me(self) -> dict: return self._req("GET", "/admin/me")
    def log(self) -> list[dict]: return self._req("GET", "/admin/log")
    def log_csv(self) -> bytes: return self._req("GET", "/admin/log.csv")
    def dashboard(self) -> dict: return self._req("GET", "/admin/dashboard")
    def approvals(self) -> list[dict]: return self._req("GET", "/admin/approvals")
    def decide(self, tid: str, decision: str, edited_reply: str | None = None, note: str | None = None) -> dict:
        return self._req("POST", f"/admin/approvals/{tid}", json={"decision": decision, "edited_reply": edited_reply, "note": note})
    def escalations(self) -> list[dict]: return self._req("GET", "/admin/escalations")
    def resolve(self, tid: str, mode: str, note: str = "", notify_customer: bool = True) -> dict:
        return self._req("POST", f"/admin/tickets/{tid}/resolve",
                         json={"mode": mode, "note": note, "notify_customer": notify_customer})
    def ticket_retrieval(self, tid: str) -> dict: return self._req("GET", f"/admin/tickets/{tid}/retrieval")
    def chunks(self, product: str, q: str | None = None, k: int = 10) -> list[dict]:
        return self._req("GET", "/admin/chunks", params={"product": product, "q": q, "k": k})
    def graph(self, product: str, limit: int = 60) -> list[dict]:
        return self._req("GET", "/admin/graph", params={"product": product, "limit": limit})
    def products(self) -> list[dict]: return self._req("GET", "/products")
