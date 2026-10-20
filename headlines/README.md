# 📰 BBC News Top Headlines

A small Python script that fetches **https://www.bbc.com/news** and prints the top 5 headlines.
It uses `requests` to download the page and `BeautifulSoup` to parse it.
It also has an **inspect mode** that helps you fix the selector if BBC changes its page layout.

## Requirements

- Python 3.8+
- Internet access

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

## Usage

```bash
python headlines.py             # print the top 5 headlines
python headlines.py --inspect   # list headline-like tags to check the selector
```

### Example output

```
1. <headline one>
2. <headline two>
3. <headline three>
4. <headline four>
5. <headline five>
```

The headlines change throughout the day.

## How it works

1. Downloads the page with a normal browser User-Agent and a 15-second timeout.
2. Selects every `h2[data-testid="card-headline"]` element, the title of each BBC story card.
3. Collapses whitespace, skips empty and duplicate headlines, and keeps page order.
4. Prints the first 5.

## When BBC changes its layout

If you see `No headlines found`, run:

```bash
python headlines.py --inspect
```

It prints the first 15 `h1`/`h2`/`h3` elements with their `class`, `data-testid` and `id`:

```
<h2 {'data-testid': 'card-headline', 'class': [...]}>  <headline text>
```

Find the tag that holds the story titles and update `HEADLINE_SELECTOR` at the top of the script.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `URL` | `https://www.bbc.com/news` | Page to read |
| `HOW_MANY` | 5 | Number of headlines printed |
| `HEADLINE_SELECTOR` | `h2[data-testid="card-headline"]` | CSS selector for headlines |

## Messages

| Message | Meaning |
|---|---|
| `Could not reach ...` | Network, DNS or timeout problem |
| `Request failed: ... HTTP 403/404` | The site blocked or moved the page |
| `No headlines found` | The selector no longer matches; use `--inspect` |
| `Only N headline(s) found` | Fewer than 5 matched on the page |

## Note

This reads the public BBC News page for personal use. Run it occasionally rather than in a tight
loop, and respect BBC's terms of use. For anything regular, BBC's official RSS feeds are a more
stable source.

## Project layout

```
.
├── headlines.py
├── requirements.txt
└── README.md
```
