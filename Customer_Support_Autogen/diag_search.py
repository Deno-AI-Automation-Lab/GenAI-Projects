"""Search diagnostics: tries each ddgs engine on its own and prints the real error.

Run from the project folder:  python diag_search.py
"""

import importlib.metadata as md
import platform
import sys
import traceback

from ddgs import DDGS

QUERY = "Tamil box office collection"

print(f"Python {sys.version.split()[0]} on {platform.platform()}")
for pkg in ("ddgs", "primp", "httpx", "lxml"):
    try:
        print(f"{pkg:6} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg:6} NOT INSTALLED")
print()

CHECKS = [
    ("text", "duckduckgo", None),
    ("text", "bing", None),
    ("text", "brave", None),
    ("text", "yahoo", None),
    ("text", "auto", None),
    ("news", "duckduckgo", "w"),
    ("news", "bing", "w"),
    ("news", "yahoo", "w"),
    ("news", "duckduckgo", None),
]

for kind, backend, tl in CHECKS:
    label = f"{kind:4} {backend:10} timelimit={tl or 'any':3}"
    for region in ("in-en", "us-en"):
        try:
            with DDGS(timeout=15) as d:
                fn = d.news if kind == "news" else d.text
                res = fn(QUERY, region=region, timelimit=tl, backend=backend, max_results=3)
            first = (res[0].get("title") or "")[:60] if res else ""
            print(f"OK    {label} region={region}: {len(res)} results  {first}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {label} region={region}: {type(exc).__name__}: {str(exc)[:160]}")

print("\nIf every line says FAIL with a connection/SSL/proxy error, your network or antivirus is")
print("blocking the search sites. Full traceback of one failure follows:\n")
try:
    with DDGS(timeout=15) as d:
        d.text(QUERY, backend="duckduckgo", max_results=3)
    print("(duckduckgo text worked, no traceback)")
except Exception:  # noqa: BLE001
    traceback.print_exc()
