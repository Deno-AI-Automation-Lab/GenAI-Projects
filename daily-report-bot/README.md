# 📰 Daily Tamil News Report Bot (PyAutoGUI + Excel)

A Windows desktop RPA bot that opens **Tamil OneIndia** in Chrome, captures up to 5 news headlines,
writes them into a formatted **Excel report** with a date/time and a category comment, saves a
**screenshot of the Excel window**, and then closes Chrome. Everything is saved to your Desktop.

It drives the real desktop (keyboard, clipboard, windows) the way a person would, with API-based
fallbacks so a report is still produced when a UI step doesn't work.

---

## ✨ What it does

1. Opens a **new Chrome window** at `https://tamil.oneindia.com/` and waits 25 s for it to load.
2. Focuses the page, presses **Ctrl+A / Ctrl+C**, and reads the clipboard (up to 3 attempts).
3. Extracts headlines from the **Breaking News** block, or falls back to Tamil headline candidates.
4. If the clipboard capture finds fewer than 3 items, it falls back to an **HTTP fetch + BeautifulSoup** extraction.
5. Starts a **separate Excel instance** (it never touches workbooks you already have open), then pastes the data with Ctrl+G / Ctrl+V. If the paste doesn't land, it writes the cells through COM instead.
6. Verifies the cells, applies formatting (Nirmala UI font for Tamil, borders, wrap, header colour, zoom to fit) and saves a real `.xlsx` file.
7. Takes a screenshot of the **actual visible Excel window**.
8. Closes the bot's Chrome window and leaves Excel open for review.

### Report columns

| Date & Time | Fetched News | Comment |
|---|---|---|
| 2026-10-20 10:30:05 | *(Tamil headline, copied verbatim)* | News on politics |

Comments are simple **keyword classifications**: politics, weather, sports or general Tamil news.
`; review headline selection` is added when the fallback headline-candidate logic was used.

---

## 📋 Requirements

- **Windows 10 / 11**
- **Google Chrome** (standard install path)
- **Desktop Microsoft Excel** (COM automation; Excel Online won't work)
- Python **3.9+**
- An **unlocked desktop** and a **primary monitor**, because the bot needs real focus and a visible screen

---

## 🚀 Setup

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Run

```bat
python daily_report_bot.py
```

- The bot waits **5 seconds** before starting.
- **Don't touch the mouse or keyboard** while it runs.
- 🛑 **Emergency stop:** move the mouse to the **top-left corner** of the screen (PyAutoGUI fail-safe).

---

## 📁 Output (on your Desktop)

The Desktop location is found through the Windows Known Folder API, so OneDrive-redirected Desktops work too.

| File | Notes |
|---|---|
| `daily_report_YYYY_MM_DD.xlsx` | If it already exists, a timestamped name is used, so earlier runs are never overwritten |
| `Excel_Screenshot.png` | If it already exists, a timestamped name is used |
| `daily_news_bot.log` | Only written when the bot fails (full error details) |

---

## ⏰ Running it daily (Task Scheduler)

The bot needs a live, unlocked desktop session, so schedule it to run **only when you are logged on**:

```bat
schtasks /create /tn "Daily Tamil News Report" /sc daily /st 09:00 /it ^
  /tr "\"C:\path\to\.venv\Scripts\python.exe\" \"C:\path\to\daily_report_bot.py\""
```

`/it` runs it interactively in your session. It **won't** work while the screen is locked or over a minimised RDP session.

---

## ⚙️ Settings (top of the script)

| Setting | Default | Meaning |
|---|---|---|
| `URL` | `https://tamil.oneindia.com/` | News site |
| `INITIAL_LOAD_SECONDS` | 25 | Wait before the first capture |
| `MAX_HEADLINES` | 5 | Headlines per report |
| `gui.PAUSE` | 0.4 | Delay between PyAutoGUI actions (raise it on slow PCs) |

To change the categories, edit the keyword lists in `comment_for()`.

---

## 🛠️ Troubleshooting

| Message / problem | Fix |
|---|---|
| `Google Chrome was not found` | Edit `find_chrome()` with your `chrome.exe` path |
| `Chrome did not create a new window` | Close Chrome's sign-in, restore or profile dialogs, then retry |
| `Could not focus the required window` | Stop other desktop activity; close pop-ups that steal focus |
| `Fewer than 3 news items found` | Check the network, cookie or notification pop-ups, or a site layout change. No empty report is created |
| `Excel lost focus` / `must be visible on the primary monitor` | Keep Excel on the main screen and don't switch windows |
| `Chrome is still open` | A "close all tabs?" dialog appeared; close it manually |
| Tamil shows as boxes | Make sure the **Nirmala UI** font is installed (it is by default on Windows 10/11) |

---

## ⚠️ Notes

- Headlines are copied **verbatim**; the category comments are keyword-based, not AI analysis.
- Headline extraction is heuristic and depends on the site's current layout.
- Use it for personal reporting, run it at most once or twice a day, and respect the site's terms of use.

---

## 📂 Project layout

```
.
├── daily_report_bot.py
├── requirements.txt
└── README.md
```
