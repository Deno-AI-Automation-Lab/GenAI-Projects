"""Long-term memory: every ticket, agent step, approval and chat turn in SQLite.

SQLite (WAL mode) keeps the full history across restarts with zero setup and is
the source of truth for the UI; Google Sheets is the business-facing log.
Customer history (by email / order id) is fed back into planning as context.
"""
from __future__ import annotations

import json
import logging
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from .models import AgentEvent, TicketResponse, now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
  ticket_id TEXT PRIMARY KEY, status TEXT NOT NULL, customer_name TEXT, order_id TEXT,
  customer_email TEXT, complaint TEXT, channel TEXT, product TEXT, category TEXT, urgency TEXT,
  sentiment TEXT, language TEXT, summary TEXT, reply_subject TEXT, reply_body TEXT,
  confidence REAL, kb_sources TEXT DEFAULT '[]', escalation_reasons TEXT DEFAULT '[]',
  extra TEXT DEFAULT '{}', created_at TEXT, updated_at TEXT);
CREATE INDEX IF NOT EXISTS ix_tickets_email ON tickets(customer_email);
CREATE INDEX IF NOT EXISTS ix_tickets_order ON tickets(order_id);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT, seq INTEGER, agent TEXT, kind TEXT,
  content TEXT, created_at TEXT);
CREATE INDEX IF NOT EXISTS ix_events_ticket ON events(ticket_id, seq);
CREATE TABLE IF NOT EXISTS approvals (
  ticket_id TEXT PRIMARY KEY, state TEXT, decision TEXT, approver TEXT, edited_reply TEXT,
  note TEXT, created_at TEXT, decided_at TEXT);
CREATE TABLE IF NOT EXISTS chats (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT, role TEXT, content TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS processed_mail (message_id TEXT PRIMARY KEY, ticket_id TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""
_JSON_FIELDS = ("kb_sources", "escalation_reasons", "extra")
log = logging.getLogger(__name__)


class MemoryStoreError(RuntimeError):
    """Long-term memory (SQLite) could not complete an operation; the API answers 503."""


class _SafeConnection:
    """SQLite connection wrapper used by MemoryStore.

    * 'database is locked' / 'busy' (another process holds a write lock, e.g. the DB is
      open in DB Browser): retried 3 times with a short back-off, then reported.
    * Any other SQLite error (disk full, file deleted, corrupt): logged and re-raised as
      MemoryStoreError with a readable message instead of a raw sqlite3 traceback.
    """

    RETRIES = 3

    def __init__(self, conn: sqlite3.Connection):
        self._c = conn

    def _run(self, fn, *args):
        for attempt in range(self.RETRIES + 1):
            try:
                return fn(*args)
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if ("locked" in msg or "busy" in msg) and attempt < self.RETRIES:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                log.error("SQLite operation failed: %s", exc)
                raise MemoryStoreError(f"Ticket storage is temporarily unavailable ({exc})") from exc
            except sqlite3.DatabaseError as exc:
                log.error("SQLite operation failed: %s", exc)
                raise MemoryStoreError(f"Ticket storage error ({exc})") from exc

    def execute(self, sql: str, params=()):
        return self._run(self._c.execute, sql, params)

    def executescript(self, script: str):
        return self._run(self._c.executescript, script)

    def commit(self) -> None:
        self._run(self._c.commit)

    def __getattr__(self, name):          # row_factory, close, ... pass straight through
        return getattr(self._c, name)


def new_ticket_id() -> str:
    """Human-friendly unique id, e.g. TKT-20261102-7F3A9C."""
    from .models import local_tz

    return f"TKT-{datetime.now(local_tz()):%Y%m%d}-{secrets.token_hex(3).upper()}"


class MemoryStore:
    """Thread-safe SQLite repository used by the API, agents and mail worker."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
        except sqlite3.Error as exc:
            raise MemoryStoreError(f"Cannot open ticket storage at {db_path} ({exc})") from exc
        raw.row_factory = sqlite3.Row
        self._conn = _SafeConnection(raw)
        self._lock = RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after the first release (existing databases keep working)."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(tickets)")}
        for name in ("confirmation_requested_at", "customer_feedback", "access_code"):
            if name not in cols:
                self._conn.execute(f"ALTER TABLE tickets ADD COLUMN {name} TEXT")
        self._conn.commit()

    # ------------------------------ tickets ------------------------------ #
    def create_ticket(self, *, customer_name: str, order_id: str | None, complaint: str,
                      customer_email: str | None, channel: str) -> str:
        """order_id is stored only when the customer gave one (NULL otherwise). Every ticket gets a
        random access code that, with the ticket ID, opens it in 'My tickets'."""
        tid, now = new_ticket_id(), now_iso()
        code = secrets.token_hex(3).upper()
        with self._lock:
            self._conn.execute(
                "INSERT INTO tickets(ticket_id,status,customer_name,order_id,customer_email,complaint,"
                "channel,created_at,updated_at,access_code) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (tid, "Processing", customer_name, order_id or None, customer_email, complaint, channel, now, now,
                 code))
            self._conn.commit()
        return tid

    def update_ticket(self, ticket_id: str, **fields: Any) -> None:
        if not fields:
            return
        for f in _JSON_FIELDS:
            if f in fields and not isinstance(fields[f], str):
                fields[f] = json.dumps(fields[f], ensure_ascii=False)
        fields["updated_at"] = now_iso()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE tickets SET {cols} WHERE ticket_id=?", (*fields.values(), ticket_id))
            self._conn.commit()

    def _row_to_ticket(self, row: sqlite3.Row) -> TicketResponse:
        d = dict(row)
        for f in ("kb_sources", "escalation_reasons"):
            d[f] = json.loads(d.get(f) or "[]")
        d.pop("extra", None)
        return TicketResponse(**d)

    def get_ticket(self, ticket_id: str) -> TicketResponse | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
        return self._row_to_ticket(row) if row else None

    def get_extra(self, ticket_id: str) -> dict:
        with self._lock:
            row = self._conn.execute("SELECT extra FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
        return json.loads(row["extra"] or "{}") if row else {}

    def list_tickets(self, limit: int = 500) -> list[TicketResponse]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM tickets ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_ticket(r) for r in rows]

    def tickets_with_status(self, status: str) -> list[TicketResponse]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM tickets WHERE status=?", (status,)).fetchall()
        return [self._row_to_ticket(r) for r in rows]

    def customer_history(self, *, email: str | None, order_id: str | None, exclude: str, limit: int = 5) -> list[dict]:
        """Previous tickets of the same customer/order – long-term memory for planning."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT ticket_id, created_at, summary, status, reply_body FROM tickets "
                "WHERE ticket_id != ? AND ((customer_email IS NOT NULL AND customer_email = ?) "
                "OR (order_id IS NOT NULL AND order_id = ?)) "
                "ORDER BY created_at DESC LIMIT ?", (exclude, email or "", order_id or "", limit)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------ events ------------------------------- #
    def add_event(self, ticket_id: str, agent: str, kind: str, content: str) -> AgentEvent:
        with self._lock:
            seq = self._conn.execute("SELECT COALESCE(MAX(seq),0)+1 FROM events WHERE ticket_id=?",
                                     (ticket_id,)).fetchone()[0]
            ev = AgentEvent(ticket_id=ticket_id, seq=seq, agent=agent, kind=kind, content=content)
            self._conn.execute("INSERT INTO events(ticket_id,seq,agent,kind,content,created_at) VALUES(?,?,?,?,?,?)",
                               (ticket_id, seq, agent, kind, content, ev.created_at))
            self._conn.commit()
        return ev

    def list_events(self, ticket_id: str, after_seq: int = 0) -> list[AgentEvent]:
        with self._lock:
            rows = self._conn.execute("SELECT ticket_id,seq,agent,kind,content,created_at FROM events "
                                      "WHERE ticket_id=? AND seq>? ORDER BY seq", (ticket_id, after_seq)).fetchall()
        return [AgentEvent(**dict(r)) for r in rows]

    # ----------------------------- approvals ----------------------------- #
    def request_approval(self, ticket_id: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO approvals(ticket_id,state,created_at) VALUES(?,?,?)",
                               (ticket_id, "pending", now_iso()))
            self._conn.commit()

    def decide_approval(self, ticket_id: str, decision: str, approver: str,
                        edited_reply: str | None, note: str | None) -> bool:
        """Record a human decision. Returns False if nothing was pending."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE approvals SET state='decided', decision=?, approver=?, edited_reply=?, note=?, decided_at=? "
                "WHERE ticket_id=? AND state='pending'",
                (decision, approver, edited_reply, note, now_iso(), ticket_id))
            self._conn.commit()
            return cur.rowcount == 1

    def get_approval(self, ticket_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM approvals WHERE ticket_id=?", (ticket_id,)).fetchone()
        return dict(row) if row else None

    def claim_approval(self, ticket_id: str) -> bool:
        """Atomically move decided -> consumed. Exactly one finalizer wins (agent or API)."""
        with self._lock:
            cur = self._conn.execute("UPDATE approvals SET state='consumed' WHERE ticket_id=? AND state='decided'",
                                     (ticket_id,))
            self._conn.commit()
            return cur.rowcount == 1

    def pending_approvals(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT ticket_id FROM approvals WHERE state='pending'").fetchall()
        return [r[0] for r in rows]

    # ------------------------------- chat -------------------------------- #
    def add_chat(self, ticket_id: str, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO chats(ticket_id,role,content,created_at) VALUES(?,?,?,?)",
                               (ticket_id, role, content, now_iso()))
            self._conn.commit()

    def chat_history(self, ticket_id: str) -> list[dict[str, str]]:
        with self._lock:
            rows = self._conn.execute("SELECT role, content FROM chats WHERE ticket_id=? ORDER BY id",
                                      (ticket_id,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------- mail -------------------------------- #
    def mail_seen(self, message_id: str) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM processed_mail WHERE message_id=?", (message_id,)).fetchone() is not None

    def mark_mail(self, message_id: str, ticket_id: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR IGNORE INTO processed_mail VALUES(?,?,?)", (message_id, ticket_id, now_iso()))
            self._conn.commit()

    # -------------------------------- kv --------------------------------- #
    def get_kv(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_kv(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv(key,value) VALUES(?,?)", (key, value))
            self._conn.commit()

    # ------------------------------ cleanup ------------------------------ #
    def tickets_by_channel(self, channel: str) -> list[TicketResponse]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM tickets WHERE channel=? ORDER BY created_at",
                                      (channel,)).fetchall()
        return [self._row_to_ticket(r) for r in rows]

    def delete_ticket(self, ticket_id: str) -> None:
        """Remove a ticket and everything attached to it (processed_mail is kept so the
        same mail is never turned into a ticket again)."""
        with self._lock:
            for table in ("tickets", "events", "approvals", "chats"):
                self._conn.execute(f"DELETE FROM {table} WHERE ticket_id=?", (ticket_id,))
            self._conn.commit()
