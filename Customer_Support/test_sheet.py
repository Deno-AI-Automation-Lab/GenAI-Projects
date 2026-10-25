"""Quick check of the Google Sheets logging setup.

Run from the project folder:   python test_sheet.py
"""
import json
from pathlib import Path

import config

print("=== Google Sheets logging check ===")
print(f"GSHEET_ENABLED              : {config.GSHEET_ENABLED}")
print(f"GSHEET_ID                   : {config.GSHEET_ID or '(EMPTY!)'}")
print(f"GSHEET_WORKSHEET (tab name) : {config.GSHEET_WORKSHEET}")
print(f"GOOGLE_SERVICE_ACCOUNT_FILE : {config.GOOGLE_SERVICE_ACCOUNT_FILE or '(EMPTY!)'}")

if not config.GSHEET_ENABLED:
    raise SystemExit("\n❌ GSHEET_ENABLED is not 'true' in .env")
if not config.GSHEET_ID:
    raise SystemExit("\n❌ GSHEET_ID is empty in .env")

key_file = Path(config.GOOGLE_SERVICE_ACCOUNT_FILE)
if not key_file.exists():
    folder = key_file.parent
    jsons = [p.name for p in folder.glob("*.json*")] if folder.exists() else []
    raise SystemExit(
        f"\n❌ Key file not found: {key_file}\n"
        f"   JSON files in that folder: {jsons or 'none'}\n"
        "   Tip: Windows may have saved it as 'service_account.json.json' "
        "(turn on View > File name extensions in Explorer).")

client_email = json.loads(key_file.read_text(encoding="utf-8")).get("client_email")
print(f"Service account email       : {client_email}")

import gspread  # noqa: E402

from sheet_logger import SheetLogger  # noqa: E402

try:
    logger = SheetLogger(config.GSHEET_ID, config.GSHEET_WORKSHEET, str(key_file))
    logger.log(case_id="TEST", turn=0, query="Connection test from test_sheet.py",
               customer_details="-", error="")
    print(f"\n✅ Success! A test row was added to tab '{config.GSHEET_WORKSHEET}'.")
    print(f"   https://docs.google.com/spreadsheets/d/{config.GSHEET_ID}/edit")
except gspread.exceptions.SpreadsheetNotFound:
    print("\n❌ Spreadsheet not found. Either GSHEET_ID is wrong, or the sheet is NOT shared with:\n"
          f"   {client_email}  (as Editor)")
except gspread.exceptions.APIError as e:
    msg = str(e)
    print(f"\n❌ Google API error:\n{msg}\n")
    if "has not been used" in msg or "is disabled" in msg or "SERVICE_DISABLED" in msg:
        print("👉 Enable 'Google Sheets API' AND 'Google Drive API' in the SAME Cloud project as the "
              "service account (APIs & Services > Library), wait 1-2 minutes, then retry.")
    elif "PERMISSION_DENIED" in msg or "403" in msg:
        print(f"👉 Share the sheet with {client_email} as **Editor** (not Viewer/Commenter).")
except PermissionError as e:
    # gspread 6 hides Google's 403 reason inside the chained exception
    cause = e.__cause__ or e.__context__
    detail = str(cause) if cause else "(no detail)"
    print(f"\n❌ PermissionError - Google refused access.\nGoogle said: {detail}\n")
    if "SERVICE_DISABLED" in detail or "has not been used" in detail or "is disabled" in detail:
        print("👉 Enable 'Google Sheets API' and 'Google Drive API' in the service account's project, "
              "wait 2 minutes, retry.")
    else:
        print(f"👉 Open the sheet > Share, and check this EXACT email is listed as Editor:\n   {client_email}\n"
              "   (copy it from service_account.json -> client_email). Also make sure GSHEET_ID is the "
              "sheet you shared.")
except Exception as e:
    cause = e.__cause__ or e.__context__
    print(f"\n❌ {type(e).__name__}: {e}" + (f"\nCause: {cause}" if cause else ""))
