"""Ticket log: Google Sheets (primary) + local CSV mirror (always, used as fallback).

Google Sheets auth uses a service-account JSON; share the sheet (owned by the
support account) with the service-account email as Editor.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from threading import Lock

from .config import Settings
from .models import TicketLogRow

log = logging.getLogger(__name__)
_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


class TicketLogger:
    """Append/update log rows in Google Sheets and a CSV mirror."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.csv_path: Path = settings.log_csv_path
        self._lock = Lock()
        self._ws = None

    # ------------------------------- sheets ------------------------------ #
    def _worksheet(self):
        """Lazily open the worksheet; returns None when Sheets is not configured."""
        if self._ws is not None:
            return self._ws
        s = self.settings
        if not (s.google_sa_json and s.google_sa_json.exists() and s.google_sheet_id):
            return None
        import gspread
        from google.oauth2.service_account import Credentials

        creds = Credentials.from_service_account_file(str(s.google_sa_json), scopes=_SCOPES)
        sh = gspread.authorize(creds).open_by_key(s.google_sheet_id)
        try:
            ws = sh.worksheet(s.google_sheet_tab)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=s.google_sheet_tab, rows=1000, cols=len(TicketLogRow.columns()))
        if ws.row_values(1) != TicketLogRow.columns():
            ws.update(range_name="A1", values=[TicketLogRow.columns()])
        self._ws = ws
        return ws

    def upsert(self, row: TicketLogRow) -> str:
        """Write the row (update if ticket_id exists). Returns where it was stored."""
        self._upsert_csv(row)
        try:
            ws = self._worksheet()
            if ws is None:
                return "csv"
            values = [getattr(row, c) for c in TicketLogRow.columns()]
            cell = ws.find(row.ticket_id, in_column=1)
            if cell:
                ws.update(range_name=f"A{cell.row}", values=[values])
            else:
                ws.append_row(values, value_input_option="RAW")
            return "google_sheets+csv"
        except Exception as exc:  # network/auth issue -> CSV already has the row
            log.error("Google Sheets write failed, CSV fallback used: %s", exc)
            self._ws = None
            return f"csv (sheets error: {exc.__class__.__name__})"

    # -------------------------------- csv -------------------------------- #
    def _upsert_csv(self, row: TicketLogRow) -> None:
        with self._lock:
            rows = self.read_csv()
            rows = [r for r in rows if r.get("ticket_id") != row.ticket_id] + [row.model_dump()]
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.csv_path.with_suffix(".tmp")
            with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=TicketLogRow.columns())
                w.writeheader()
                w.writerows(rows)
            tmp.replace(self.csv_path)  # atomic swap

    def read_csv(self) -> list[dict]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open(newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))

    def delete_tickets(self, ticket_ids: set[str]) -> tuple[int, int]:
        """Remove rows for these ticket IDs from the CSV and the Google Sheet. Returns (csv, sheet) counts."""
        with self._lock:
            rows = self.read_csv()
            keep = [r for r in rows if r.get("ticket_id") not in ticket_ids]
            if len(keep) != len(rows):
                with self.csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
                    w = csv.DictWriter(fh, fieldnames=TicketLogRow.columns())
                    w.writeheader()
                    w.writerows(keep)
        sheet_removed = 0
        ws = self._worksheet()
        if ws is not None:
            col = ws.col_values(1)
            # delete bottom-up so row numbers stay valid
            for idx in sorted((i + 1 for i, v in enumerate(col) if v in ticket_ids), reverse=True):
                ws.delete_rows(idx)
                sheet_removed += 1
        return len(rows) - len(keep), sheet_removed

    def read_all(self) -> list[dict]:
        """Rows for the log viewer: Google Sheet when reachable, else CSV."""
        try:
            ws = self._worksheet()
            if ws is not None:
                return ws.get_all_records()
        except Exception as exc:
            log.warning("Sheets read failed, using CSV: %s", exc)
        return self.read_csv()
