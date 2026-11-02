"""Show a ticket's status, customer email, approval state and full agent timeline.

    python scripts/show_ticket.py TKT-20261102-ABC123
    python scripts/show_ticket.py --last          # most recent ticket
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from support_agent.llm import bootstrap_runtime  # noqa: E402
from support_agent.services import build_services  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticket_id", nargs="?")
    ap.add_argument("--last", action="store_true")
    args = ap.parse_args()
    bootstrap_runtime()
    svc = build_services()
    if args.last or not args.ticket_id:
        recent = svc.memory.list_tickets(1)
        if not recent:
            print("No tickets yet.")
            return
        tid = recent[0].ticket_id
    else:
        tid = args.ticket_id
    t = svc.memory.get_ticket(tid)
    if t is None:
        print(f"{tid} not found")
        return
    print(f"Ticket   : {t.ticket_id}   status: {t.status}   channel: {t.channel}")
    print(f"Customer : {t.customer_name}   email: {t.customer_email or '(none - UI only)'}   order: {t.order_id}")
    print(f"Triage   : {t.category} / {t.urgency} / {t.sentiment}   confidence: {t.confidence}")
    print(f"Reasons  : {'; '.join(t.escalation_reasons) or '-'}")
    print(f"Approval : {svc.memory.get_approval(tid)}")
    print(f"Subject  : {t.reply_subject}\n")
    for e in svc.memory.list_events(tid):
        print(f"[{e.seq:>3}] {e.created_at}  {e.agent:<22} {e.kind:<11} {e.content[:160].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
