"""Print the top 5 headlines from BBC News.

Usage:
    python headlines.py           # print the top 5 headlines
    python headlines.py --inspect # show which tags hold headline-like text,
                                  # so you can confirm the selector yourself
"""

import sys

import requests
from bs4 import BeautifulSoup

URL = "https://www.bbc.com/news"
HOW_MANY = 5

# A normal browser User-Agent; some sites refuse the default python-requests one.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}

# On bbc.com/news each story card's title sits in:
#     <h2 data-testid="card-headline">Headline text</h2>
# If BBC changes its markup, run with --inspect and update this selector.
HEADLINE_SELECTOR = 'h2[data-testid="card-headline"]'


def fetch_page(url):
    """Download the page and return its HTML, or exit with a clear message."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
    except requests.RequestException as err:
        sys.exit(f"Could not reach {url}: {err}")

    if response.status_code != 200:
        sys.exit(f"Request failed: {url} returned HTTP {response.status_code}")

    return response.text


def extract_headlines(html, limit):
    """Return up to `limit` unique, non-empty headline strings in page order."""
    soup = BeautifulSoup(html, "html.parser")
    headlines = []
    seen = set()

    for tag in soup.select(HEADLINE_SELECTOR):
        text = " ".join(tag.get_text().split())  # strip and collapse whitespace
        if not text or text in seen:
            continue
        seen.add(text)
        headlines.append(text)
        if len(headlines) == limit:
            break

    return headlines


def inspect(html):
    """Print the tags and attributes of headline-like elements on the page."""
    soup = BeautifulSoup(html, "html.parser")
    print("First 15 h1/h2/h3 elements on the page:\n")
    for tag in soup.find_all(["h1", "h2", "h3"])[:15]:
        attrs = {k: v for k, v in tag.attrs.items() if k in ("class", "data-testid", "id")}
        text = " ".join(tag.get_text().split())[:70]
        print(f"<{tag.name} {attrs}>  {text}")


def main():
    html = fetch_page(URL)

    if "--inspect" in sys.argv:
        inspect(html)
        return

    headlines = extract_headlines(html, HOW_MANY)

    if not headlines:
        print("No headlines found. The page layout may have changed; "
              "run `python headlines.py --inspect` to find the new tag.")
        return

    for number, headline in enumerate(headlines, start=1):
        print(f"{number}. {headline}")

    if len(headlines) < HOW_MANY:
        print(f"\n(Only {len(headlines)} headline(s) found, expected {HOW_MANY}.)")


if __name__ == "__main__":
    main()