# 🏏 Cricbuzz Live Score Scraper (Playwright)

A small Playwright script that opens the Cricbuzz home page in Chromium, reads the match/score rows,
prints them as clean single lines, and saves a screenshot of the page.

## What it does

1. Launches Chromium in **visible** mode (`headless=False`) so you can watch it work.
2. Opens `https://www.cricbuzz.com/`.
3. Waits for the score elements (`span.font-medium.text-cbTxtPrim.truncate`) to load.
4. Reads the parent row of every non-empty match, so team names and scores stay together.
5. Collapses extra spaces and line breaks, then prints one line per row.
6. Saves a full-window screenshot as `score.png`.

## Example output

```
IND 285/6 (50) AUS 190/4 (32.1)
ENG 312/8 (50) NZ 298 (49.2)
Saved score.png
```

(Actual rows depend on the matches showing on Cricbuzz when you run it.)

## Requirements

- Python 3.9+
- Internet access

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
python -m playwright install chromium
```

## Run

```bash
python cricbuzz_score.py
```

A Chromium window opens briefly, the rows print in the terminal, and `score.png` is saved in the current folder.

## Customise

| Change | How |
|---|---|
| Run without a visible browser (servers, scheduled jobs) | `p.chromium.launch(headless=True)` |
| Capture the whole page, not just the window | `page.screenshot(path="score.png", full_page=True)` |
| Allow more time for slow networks | `page.wait_for_selector(selector, timeout=60_000)` |

## Troubleshooting

| Problem | Fix |
|---|---|
| `Executable doesn't exist` | Run `python -m playwright install chromium` |
| `Timeout ... waiting for selector` | Cricbuzz has changed its page layout. Inspect a score in the browser (F12) and update `selector` |
| No rows printed | No matches are listed right now, or the class names have changed |

## Note

This script reads the public Cricbuzz page for personal use. Use it sparingly, respect Cricbuzz's
terms of use, and don't run it in a tight loop. CSS class names like `text-cbTxtPrim` can change
without notice, so expect to update the selector from time to time.

## Project layout

```
.
├── cricbuzz_score.py
├── requirements.txt
├── README.md
└── score.png      # generated screenshot
```
