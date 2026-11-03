"""Export a shareable sample of the REAL ticket log for the submission.

storage/ is not submitted (it holds the database and private data), so this writes
docs/sample_ticket_log.csv: the same 12 columns as the Google Sheet / CSV log, from your
own runs, with personal data removed:

* rows whose customer name you list with --exclude-names are dropped (e.g. your own tests)
* e-mail addresses and phone numbers anywhere in the text are masked
* by default the newest rows per status are kept, so every status is shown

    python scripts/export_sample_log.py --exclude-names "Deno Melvin,Melvin" --per-status 3

Open the file and check it before you submit.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from support_agent.config import get_settings  # noqa: E402
from support_agent.models import TicketLogRow  # noqa: E402

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
# 10+ digit phone numbers such as +91 98765 43210 or 9876543210 (not dates, times or order IDs)
PHONE = re.compile(r"(?<![\w-])\+?\d{1,3}[\s-]?\d{4,5}[\s-]?\d{4,6}(?![\w-])")
TEXT_COLUMNS = {"customer_name", "complaint", "response", "escalation_reason"}   # free text only


def mask(text: str) -> str:
    return PHONE.sub("[phone]", EMAIL.sub("[email]", text or ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-names", default="", help="comma-separated customer names to leave out")
    ap.add_argument("--per-status", type=int, default=3, help="newest rows kept per status (0 = all)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "sample_ticket_log.csv"))
    args = ap.parse_args()

    src = get_settings().log_csv_path
    if not src.exists():
        raise SystemExit(f"No ticket log at {src}. Raise a few tickets first.")
    with src.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    excluded = {n.strip().lower() for n in args.exclude_names.split(",") if n.strip()}
    rows = [r for r in rows if (r.get("customer_name") or "").strip().lower() not in excluded]
    rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    if args.per_status:
        by_status: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if len(by_status[r.get("status", "")]) < args.per_status:
                by_status[r.get("status", "")].append(r)
        rows = sorted((r for v in by_status.values() for r in v), key=lambda r: r.get("timestamp", ""))

    cols = TicketLogRow.columns()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:   # utf-8-sig: Excel shows Tamil/Hindi correctly
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: mask(r.get(c, "")) if c in TEXT_COLUMNS else r.get(c, "") for c in cols})
    statuses = sorted({r.get("status", "") for r in rows})
    print(f"Wrote {len(rows)} rows to {out}")
    print("Statuses included: " + ", ".join(statuses))
    print("Open the file and check it contains no personal data before submitting.")


if __name__ == "__main__":
    main()
