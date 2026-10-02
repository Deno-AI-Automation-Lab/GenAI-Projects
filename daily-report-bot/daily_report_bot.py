"""Windows 10/11 daily news bot (Chrome + desktop Microsoft Excel required).

Install:
    python -m pip install pyautogui pyperclip pillow pywin32 requests beautifulsoup4
Run:
    python daily_news_bot.py

Keep the desktop unlocked and do not use the mouse/keyboard while running.
Move the mouse to the upper-left corner to trigger PyAutoGUI's emergency stop.
PyAutoGUI controls Chrome, copies the page, pastes into Excel and screenshots it.
pywin32 starts Excel, formats cells and saves a real .xlsx file. HTTP extraction is a fallback if clipboard capture fails.
The captured news is copied verbatim; comments are simple keyword classifications.
"""

import ctypes
from ctypes import wintypes
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
import logging

import requests
from bs4 import BeautifulSoup

import pyautogui as gui
import pyperclip
import win32com.client
import win32gui
import win32con

URL = "https://tamil.oneindia.com/"
INITIAL_LOAD_SECONDS = 25
LOAD_TIMEOUT_SECONDS = 90
MAX_HEADLINES = 5

gui.FAILSAFE = True
gui.PAUSE = 0.4


def desktop_path():
    # Windows Known Folder handles redirected/OneDrive Desktop locations.
    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]
    folder_id = GUID.from_buffer_copy(
        uuid.UUID("B4BFCC3A-DB2C-424C-B029-7FE99A87C641").bytes_le)
    result = ctypes.c_wchar_p()
    shell = ctypes.windll.shell32
    shell.SHGetKnownFolderPath.argtypes = [ctypes.POINTER(GUID), wintypes.DWORD,
                                         wintypes.HANDLE, ctypes.POINTER(ctypes.c_wchar_p)]
    shell.SHGetKnownFolderPath.restype = ctypes.c_long
    if shell.SHGetKnownFolderPath(ctypes.byref(folder_id), 0, None,
                                  ctypes.byref(result)) != 0:
        raise RuntimeError("Could not resolve the Windows Desktop folder.")
    try:
        return Path(result.value)
    finally:
        ctypes.windll.ole32.CoTaskMemFree(ctypes.cast(result, ctypes.c_void_p))


def find_chrome():
    for base in (os.getenv("PROGRAMFILES"), os.getenv("PROGRAMFILES(X86)"),
                 os.getenv("LOCALAPPDATA")):
        if base:
            path = Path(base) / "Google/Chrome/Application/chrome.exe"
            if path.is_file():
                return str(path)
    raise RuntimeError("Google Chrome was not found. Edit find_chrome() for your installation.")


def activate_window(hwnd):
    win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    # An Alt key event helps Windows allow foreground activation.
    gui.press("alt")
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(1)
    if win32gui.GetForegroundWindow() != hwnd:
        raise RuntimeError("Could not focus the required window. Stop other desktop activity.")


def chrome_windows():
    matches = set()
    def visit(hwnd, _):
        if (win32gui.IsWindowVisible(hwnd)
                and win32gui.GetClassName(hwnd) == "Chrome_WidgetWin_1"):
            matches.add(hwnd)
    win32gui.EnumWindows(visit, None)
    return matches


def open_chrome():
    previous = chrome_windows()
    subprocess.Popen([find_chrome(), "--new-window", URL])
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        created = chrome_windows() - previous
        if created:
            return next(iter(created))
        time.sleep(1)
    raise RuntimeError("Chrome did not create a new window. Check Chrome startup dialogs.")


def copy_page(hwnd):
    activate_window(hwnd)
    gui.press("esc")
    gui.hotkey("ctrl", "home")
    # Move focus from address bar into the page without a coordinate click.
    gui.hotkey("ctrl", "l")
    gui.press("esc")
    gui.hotkey("shift", "f6")
    marker = "BOT_PENDING_" + uuid.uuid4().hex
    pyperclip.copy(marker)
    gui.hotkey("ctrl", "a")
    gui.hotkey("ctrl", "c")
    time.sleep(1)
    value = pyperclip.paste()
    gui.press("right")
    if value == marker:
        return ""
    return value


def extract_news(text):
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    news = []
    start = next((i for i, line in enumerate(lines)
                  if line.casefold() in {"breaking news", "பிரேக்கிங் நியூஸ்",
                                         "பிரேக்கிங் செய்திகள்"}), None)
    if start is not None:
        for line in lines[start + 1:start + 25]:
            if line in {"செய்திகள்", "News", "Read Full Story", "Read Story"}:
                break
            if len(line) >= 25 and re.search(r"[\u0b80-\u0bff]", line):
                if line not in news:
                    news.append(line)
            if len(news) >= MAX_HEADLINES:
                break
    breaking_count = len(news)
    if len(news) >= 3:
        return news, "breaking"
    # A heuristic fallback, not a semantic guarantee of headline selection.
    for line in lines:
        if (40 <= len(line) <= 500 and re.search(r"[\u0b80-\u0bff]", line)
                and not line.startswith(("Copyright", "©")) and line not in news):
            news.append(line)
        if len(news) >= MAX_HEADLINES:
            break
    return news, "mixed headline candidates" if breaking_count else "headline candidates"


def http_headlines():
    # Helper fallback: never saves navigation text or an error page as news.
    response = requests.get(URL, timeout=30, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130.0 Safari/537.36"})
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    headlines = []
    for link in soup.select("a[href]"):
        href = link.get("href", "")
        text = re.sub(r"\s+", " ", link.get_text(" ", strip=True))
        text = re.sub(r"^(Read Full Story|Read Story)\s*", "", text)
        if (re.search(r"/[^/]+-\d+\.html(?:[?#].*)?$", href)
                and (href.startswith("/") or href.startswith(URL))
                and 30 <= len(text) <= 500
                and re.search(r"[\u0b80-\u0bff]", text)
                and text not in headlines):
            headlines.append(text)
        if len(headlines) == MAX_HEADLINES:
            break
    return headlines, "news headlines"


def fetch_news(hwnd):
    print(f"Waiting {INITIAL_LOAD_SECONDS}s for the page to load...", flush=True)
    time.sleep(INITIAL_LOAD_SECONDS)
    for attempt in range(3):
        headlines, source = extract_news(copy_page(hwnd))
        print(f"Clipboard attempt {attempt + 1}: {len(headlines)} news items", flush=True)
        # Live tickers can change continuously; identical snapshots are not required.
        if len(headlines) >= 3:
            return headlines[:MAX_HEADLINES], source
        time.sleep(5)
    print("Clipboard capture insufficient; trying HTTP headline helper...", flush=True)
    headlines, source = http_headlines()
    if len(headlines) < 3:
        raise RuntimeError("Fewer than 3 news items found. Check pop-ups/network or site layout. "
                           "No empty report was created.")
    return headlines[:MAX_HEADLINES], source


def comment_for(headlines, source):
    text = " ".join(headlines)
    if any(word in text for word in ("அரசியல்", "திமுக", "அதிமுக", "தேர்தல்", "முதல்வர்", "பிரதமர்")):
        category = "News on politics"
    elif any(word in text for word in ("மழை", "வானிலை", "புயல்")):
        category = "News on weather"
    elif any(word in text for word in ("கிரிக்கெட்", "விளையாட்டு", "கால்பந்து")):
        category = "News on sports"
    else:
        category = "General Tamil news"
    return category + ("; review headline selection" if "candidates" in source else "")


def paste_range(address, value):
    gui.hotkey("ctrl", "g")
    gui.write(address, interval=0.1)
    gui.press("enter")
    pyperclip.copy(value)
    gui.hotkey("ctrl", "v")
    time.sleep(0.8)
    gui.press("esc")


def safe_cell(text):
    # No tabs/newlines in TSV cells; prevent pasted headlines becoming formulas.
    text = re.sub(r"[\r\n\t]+", " ", text).strip()
    return "'" + text if text.startswith(("=", "+", "-", "@")) else text


def save_report(headlines, source):
    now = datetime.now()  # Local computer date/time, generated at run time.
    desktop = desktop_path()
    report = desktop / f"daily_report_{now:%Y_%m_%d}.xlsx"
    screenshot = desktop / "Excel_Screenshot.png"
    if screenshot.exists():
        screenshot = desktop / f"Excel_Screenshot_{now:%Y%m%d_%H%M%S_%f}.png"

    # Dedicated Excel process; no interference with an already-open workbook.
    excel = win32com.client.DispatchEx("Excel.Application")
    excel.Visible = True
    excel.DisplayAlerts = True
    # Each run contains at most five entries. Preserve earlier runs using a
    # timestamp suffix instead of appending indefinitely or overwriting files.
    if report.exists():
        report = desktop / f"daily_report_{now:%Y_%m_%d_%H%M%S_%f}.xlsx"
    book = excel.Workbooks.Add()
    sheet = book.Worksheets(1)
    sheet.Name = "Daily News"
    sheet.Activate()
    activate_window(excel.Hwnd)
    rows = [["Date & Time", "Fetched News", "Comment"]]
    for headline in headlines[:MAX_HEADLINES]:
        rows.append([now.strftime("%Y-%m-%d %H:%M:%S"), headline,
                     comment_for([headline], source)])
    row = len(rows)
    sheet.Range(f"A1:C{row}").NumberFormat = "@"
    tsv = "\r\n".join("\t".join(safe_cell(cell) for cell in record) for record in rows)
    paste_range("A1", tsv)
    # If Excel intercepted the paste, use its native helper to ensure the report
    # is populated. PyAutoGUI still handles Chrome, paste and screenshot.
    if any(str(sheet.Cells(i, 2).Value or "") != record[1]
           for i, record in enumerate(rows, 1)):
        print("Excel clipboard paste did not complete; using Excel helper.", flush=True)
        sheet.Range(f"A1:C{row}").Value = tuple(tuple(record) for record in rows)
    if any(str(sheet.Cells(i, 2).Value or "") != record[1]
           for i, record in enumerate(rows, 1)):
        raise RuntimeError("Excel data verification failed.")

    sheet.Columns("A").ColumnWidth = 23
    sheet.Columns("B").ColumnWidth = 72
    sheet.Columns("C").ColumnWidth = 30
    area = sheet.Range(f"A1:C{row}")
    area.Font.Name = "Nirmala UI"
    area.Font.Size = 11
    area.WrapText = True
    area.VerticalAlignment = -4160  # Top
    area.Borders.LineStyle = 1
    sheet.Range("A1:C1").Font.Bold = True
    sheet.Range("A1:C1").Interior.Color = 0xEAD9C4
    sheet.Range(f"A2:C{row}").Rows.AutoFit()
    for index in range(2, row + 1):
        sheet.Rows(index).RowHeight = max(50, sheet.Rows(index).RowHeight)
    sheet.Rows(1).RowHeight = 30
    excel.ActiveWindow.Zoom = 75
    book.SaveAs(str(report), FileFormat=51)  # Genuine .xlsx format.
    if not report.is_file() or not book.Saved:
        raise RuntimeError("Excel did not save the workbook successfully.")

    activate_window(excel.Hwnd)
    paste_range_selection = sheet.Range("A1")
    excel.Goto(paste_range_selection, True)
    excel.ActiveWindow.ScrollColumn = 1
    excel.ActiveWindow.ScrollRow = 1
    # Fit all three columns and all rows into the visible Excel sheet.
    sheet.Range(f"A1:C{row}").Select()
    excel.ActiveWindow.Zoom = True
    sheet.Range("A1").Select()
    book.Save()
    time.sleep(2)
    if win32gui.GetForegroundWindow() != excel.Hwnd:
        raise RuntimeError("Excel lost focus; screenshot was not taken.")
    # Screenshot the actual visible Excel window, not a rendered substitute.
    left, top, right, bottom = win32gui.GetWindowRect(excel.Hwnd)
    width, height = gui.size()
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if right <= left or bottom <= top:
        raise RuntimeError("Excel must be visible on the primary monitor.")
    gui.screenshot(region=(left, top, right - left, bottom - top)).save(screenshot)
    if not screenshot.is_file():
        raise RuntimeError("Screenshot save failed.")
    print(f"Excel saved: {report}\nScreenshot saved: {screenshot}")
    print("Excel remains open for review.")


def main():
    if os.name != "nt":
        raise RuntimeError("This bot requires Windows and installed desktop Microsoft Excel.")
    print("Starting in 5 seconds. Leave mouse and keyboard idle.")
    time.sleep(5)
    hwnd = open_chrome()
    headlines, source = fetch_news(hwnd)
    print(f"Captured {len(headlines)} items ({source}).")
    save_report(headlines, source)
    activate_window(hwnd)
    gui.hotkey("alt", "f4")
    time.sleep(2)
    if win32gui.IsWindow(hwnd):
        raise RuntimeError("Both files saved, but Chrome is still open; check its close dialog.")
    print("Finished: both files saved and the bot Chrome window closed.", flush=True)


if __name__ == "__main__":
    log_path = desktop_path() / "daily_news_bot.log" if os.name == "nt" else Path("daily_news_bot.log")
    logging.basicConfig(filename=str(log_path), level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    try:
        main()
    except gui.FailSafeException:
        print("Stopped by PyAutoGUI emergency fail-safe.")
        raise SystemExit(1)
    except Exception as error:
        logging.exception("Bot failed")
        print(f"Bot failed: {error}\nError details: {log_path}")
        input("Press Enter to close...")
        raise SystemExit(1)