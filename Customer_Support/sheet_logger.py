"""Append every customer query (and the agent's result) as one row in a Google Sheet.

Auth: Google service account JSON key (path in .env as GOOGLE_SERVICE_ACCOUNT_FILE).
The sheet must be shared with the service account's email as Editor.
"""
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import gspread

import config

HEADERS = [
    "Timestamp (IST)", "Case ID", "Turn #", "Customer Query", "Customer Details",
    "Scope Verdict", "Scope Reason", "Agent Route", "Answer Source", "Fault Domain",
    "Priority", "Ticket Tags", "Escalation", "Escalation Team", "Agent Reply",
    "Web Sources", "Handbook Pages", "Response Time (s)", "Status", "Error",
]

_MAX_CELL = 45000  # Google Sheets limit is 50,000 characters per cell

ROUTE_LABELS = {
    "handbook": "Triage → Troubleshooter → Support Specialist",
    "hybrid": "Triage → Troubleshooter → Web Research → Support Specialist",
    "web": "Web Research → Support Specialist",
    None: "Scope check only (web search off)",
}


def _cell(value):
    if isinstance(value, (int, float)):
        return value
    text = "" if value is None else str(value)
    return text if len(text) <= _MAX_CELL else text[:_MAX_CELL] + "…[truncated]"


class SheetLogger:
    def __init__(self, spreadsheet_id: str, worksheet_name: str, credentials_file: str):
        client = gspread.service_account(filename=credentials_file)
        spreadsheet = client.open_by_key(spreadsheet_id)
        try:
            self.ws = spreadsheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            self.ws = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=len(HEADERS))
        self._ensure_headers()

    def _ensure_headers(self) -> None:
        # Only write headers into an empty sheet; never overwrite existing data
        if not self.ws.row_values(1):
            self.ws.update(range_name="A1", values=[HEADERS])
            try:
                self.ws.freeze(rows=1)
                self.ws.format("1:1", {"textFormat": {"bold": True}})
            except Exception:
                pass

    def log(self, *, case_id: str, turn: int, query: str, customer_details: str,
            decision=None, mode: Optional[str] = None, result=None, sources=None,
            elapsed_s: float = 0.0, error: str = "") -> None:
        now = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
        pages = sorted({f"p.{s['page']} – {s['section']}" for s in (sources or [])})
        row = [
            now, case_id, turn, query, customer_details,
            getattr(decision, "verdict", ""), getattr(decision, "reason", ""),
            ROUTE_LABELS.get(mode, "") if decision else "",
            getattr(result, "answer_source", ""), getattr(result, "fault_domain", ""),
            getattr(result, "priority", ""), ", ".join(getattr(result, "ticket_tags", []) or []),
            ("Yes" if result.escalation_required else "No") if result else "",
            getattr(result, "escalation_team", ""), getattr(result, "customer_reply", ""),
            "\n".join(getattr(result, "web_sources", []) or []), "\n".join(pages),
            round(elapsed_s, 1), "Error" if error else "Answered", error,
        ]
        # RAW (not USER_ENTERED) so a query starting with "=" is stored as text, never run as a formula
        self.ws.append_row([_cell(v) for v in row], value_input_option="RAW",
                           insert_data_option="INSERT_ROWS")


def get_logger() -> Optional[SheetLogger]:
    """Return a logger, or None if Google Sheets logging isn't configured."""
    if not (config.GSHEET_ENABLED and config.GSHEET_ID and config.GOOGLE_SERVICE_ACCOUNT_FILE):
        return None
    return SheetLogger(config.GSHEET_ID, config.GSHEET_WORKSHEET, config.GOOGLE_SERVICE_ACCOUNT_FILE)
