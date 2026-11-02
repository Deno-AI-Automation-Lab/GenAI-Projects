"""Remove tickets that were created from email (e.g. junk mail picked up by mistake).

Deletes them from long-term memory (SQLite), the CSV log and the Google Sheet.
Dry run by default – nothing is deleted until you add --yes.

    python scripts/purge_email_tickets.py                 # list email tickets (dry run)
    python scripts/purge_email_tickets.py --yes           # delete ALL email tickets
    python scripts/purge_email_tickets.py --ids TKT-20261102-B8C2E5 TKT-20261102-E856DC --yes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from support_agent.llm import bootstrap_runtime  # noqa: E402
from support_agent.services import build_services  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="*", help="only these ticket IDs (default: every email ticket)")
    ap.add_argument("--yes", action="store_true", help="actually delete (otherwise dry run)")
    args = ap.parse_args()

    bootstrap_runtime()
    svc = build_services()
    tickets = svc.memory.tickets_by_channel("email")
    if args.ids:
        wanted = set(args.ids)
        tickets = [t for t in tickets if t.ticket_id in wanted]
    if not tickets:
        print("No matching email tickets.")
        return
    for t in tickets:
        print(f"{t.ticket_id}  {t.created_at}  {t.customer_name[:30]:<30}  {t.complaint[:60]!r}")
    print(f"\n{len(tickets)} email ticket(s) found.")
    if not args.yes:
        print("Dry run - nothing deleted. Re-run with --yes to delete them.")
        return
    ids = {t.ticket_id for t in tickets}
    for tid in ids:
        svc.memory.delete_ticket(tid)
    csv_n, sheet_n = svc.logger.delete_tickets(ids)
    print(f"Deleted {len(ids)} ticket(s) from memory, {csv_n} CSV row(s), {sheet_n} Google Sheet row(s).")


if __name__ == "__main__":
    main()
