"""Free, keyless web search via the DuckDuckGo `ddgs` package.

Supports DuckDuckGo News (dated articles) and a time filter, so time-sensitive
questions ("running in theatres now", "forecast today") get recent results.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Annotated, Any, Literal

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from config import settings

logger = logging.getLogger("support_bot.search")

_SNIPPET_CHARS = 400

_last_error: list[str] = [""]  # most recent engine error, reported if nothing works

TimeLimit = Literal["d", "w", "m", "y", "any"]


_DEFAULT_ENGINES = {"news": "duckduckgo,bing,yahoo", "text": "duckduckgo,bing,brave,yahoo"}


def _engines(kind: str) -> list[str]:
    # getattr keeps this working with an older config.py that lacks these settings
    raw = getattr(settings, f"search_backend_{kind}", None) or _DEFAULT_ENGINES[kind]
    return [e.strip() for e in raw.split(",") if e.strip()]


def _search_sync(kind: str, query: str, max_results: int, timelimit: str | None) -> list[dict[str, Any]]:
    """Try each engine on its own, in order (DuckDuckGo first), and return the first non-empty result.

    Engines are queried one at a time on purpose: passing "duckduckgo,bing" to ddgs in a single
    call lets a fast failure from one engine end the search before a slower engine answers.
    """
    for engine in _engines(kind):
        try:
            with DDGS(timeout=15) as ddgs:
                fn = ddgs.news if kind == "news" else ddgs.text
                results = fn(
                    query,
                    region=getattr(settings, "search_region", "in-en"),
                    safesearch="moderate",
                    timelimit=timelimit,
                    backend=engine,
                    max_results=max_results,
                )
            if results:
                logger.info("%s search via %s: %d results for '%s'", kind, engine, len(results), query)
                for r in results:
                    r.setdefault("engine", engine)
                return results
        except (DDGSException, RatelimitException, TimeoutException) as exc:
            logger.info("%s search via %s failed for '%s': %s", kind, engine, query, exc)
            if "No results found" not in str(exc):
                _last_error[0] = f"{engine}: {str(exc)[:160]}"
    return []


async def _search_with_retry(kind: str, query: str, max_results: int, timelimit: str | None) -> list[dict[str, Any]]:
    """Run the engine chain off the event loop; one short retry if every engine came back empty."""
    for attempt in range(2):
        results = await asyncio.to_thread(_search_sync, kind, query, max_results, timelimit)
        if results or not _last_error[0]:
            return results
        await asyncio.sleep(1.5)  # an engine errored (rate limit / network); retry the chain once
    return []


_TIME_WORDS = re.compile(
    r"\b(now|today|tonight|currently|current|latest|recent|recently|trending|this week|this month|"
    r"right now|these days|running|showing|january|february|march|april|may|june|july|august|september|"
    r"october|november|december|20\d\d)\b",
    re.IGNORECASE,
)


def simplify_query(query: str) -> str:
    """Drop date/recency words; the timelimit filter already handles recency."""
    return re.sub(r"\s+", " ", _TIME_WORDS.sub(" ", query)).strip()


class EvidenceStore:
    """Every search result seen while answering the current question, with stable ids [S1], [S2]...

    The specialist cites these ids in its bullets and the quality observer re-reads them
    to verify each claim. Reset at the start of every question (one CLI user at a time;
    a multi-user server would keep one store per session).
    """

    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []
        self.searches: list[str] = []

    def reset(self) -> None:
        self.items.clear()
        self.searches.clear()

    def add(self, r: dict[str, Any], max_chars: int = _SNIPPET_CHARS) -> str:
        url = r.get("url") or r.get("href") or ""
        for item in self.items:
            if url and item["url"] == url:
                return item["id"]
        item = {
            "id": f"S{len(self.items) + 1}",
            "title": (r.get("title") or "").strip(),
            "body": (r.get("body") or "").strip()[:max_chars],
            "url": url,
            "date": (r.get("date") or "")[:10],
            "source": r.get("source") or "",
        }
        self.items.append(item)
        return item["id"]

    def render(self, ids: list[str] | None = None) -> str:
        rows = [i for i in self.items if ids is None or i["id"] in ids]
        out = []
        for i in rows:
            meta = " | ".join(x for x in (f"Published: {i['date']}" if i["date"] else "", i["source"]) if x)
            out.append(f"[{i['id']}] {i['title']}\n   {i['body']}\n   {meta + ' | ' if meta else ''}{i['url']}")
        return "\n".join(out)

    def url(self, source_id: str) -> str:
        return next((i["url"] for i in self.items if i["id"] == source_id), "")


EVIDENCE = EvidenceStore()


def _format(results: list[dict[str, Any]], kind: str) -> str:
    if kind == "news":  # newest first
        results = sorted(results, key=lambda r: r.get("date") or "", reverse=True)
    ids = [EVIDENCE.add(r) for r in results]
    return EVIDENCE.render(list(dict.fromkeys(ids)))


def get_search_evidence() -> str:
    """Return every web search result the specialist received for the current question.

    Results are labelled [S1], [S2], ... matching the citations in the specialist's
    bullets. Use this to check that each name, date and number in the reply is
    actually stated in the cited result.

    Returns:
        All search results for this question, or a note that no search was made.
    """
    header = f"Searches made: {len(EVIDENCE.searches)}" + "".join(f"\n  - {q}" for q in EVIDENCE.searches)
    return header + "\n\n" + (EVIDENCE.render() or "No search results were collected for this question.")


async def web_search(
    query: Annotated[str, "Short keyword query, e.g. 'Tamil box office collection'. No dates when using timelimit."],
    news: Annotated[bool, "True to search recent news articles (best for anything happening now)."] = False,
    timelimit: Annotated[TimeLimit, "Only results from the last d=day, w=week, m=month, y=year, or 'any'."] = "any",
    max_results: Annotated[int, "How many results to return (1-8). Default 5."] = 5,
) -> str:
    """Search the web with DuckDuckGo and return the top results, newest first for news.

    Use this tool for any question that needs current or factual information
    (food and nutrition facts, travel details, workouts, movies, or weather).
    For anything time-sensitive (now, today, this week, latest, upcoming,
    running in theatres, forecast, new rules) set news=True and timelimit='w'
    (or 'd' for today's events). For evergreen topics (recipes, exercises)
    leave the defaults. Each result has an id like [S3], a title, snippet,
    publish date when known, and URL. Write your own answer from the results,
    citing the id of the result each fact came from; do not paste them.

    Args:
        query: What to search for.
        news: Search DuckDuckGo News instead of the general web index.
        timelimit: Restrict results to the last day, week, month or year.
        max_results: Number of results to fetch, between 1 and 8.

    Returns:
        A plain-text list of results labelled [S1], [S2], ..., or a message explaining why
        the search returned nothing.
    """
    query = (query or "").strip()
    if not query:
        return "Search skipped: the query was empty."
    EVIDENCE.searches.append(f"'{query}' (news={news}, timelimit={timelimit})")
    max_results = max(1, min(int(max_results), 8))
    limit = None if timelimit == "any" else timelimit

    # Try the most specific search first, then widen so the agent always gets something.
    attempts: list[tuple[str, str, str | None]] = []
    for q in dict.fromkeys([query, simplify_query(query)]):  # original, then without date words
        if not q:
            continue
        if news:
            attempts.append(("news", q, limit or "w"))
            attempts.append(("news", q, "m"))
        attempts.append(("text", q, limit))
        if limit:
            attempts.append(("text", q, None))

    tried: set[tuple[str, str, str | None]] = set()
    _last_error[0] = ""
    try:
        for i, (kind, q, tl) in enumerate(attempts):
            if (kind, q, tl) in tried:
                continue
            tried.add((kind, q, tl))
            if i:
                await asyncio.sleep(0.4)  # be gentle; rapid-fire queries trigger rate limits
            results = await _search_with_retry(kind, q, max_results, tl)
            if results:
                note = "" if i == 0 else f"(Note: widened search to {kind} '{q}', timelimit={tl or 'any'}.)\n"
                return note + _format(results, kind)
    except (RatelimitException, TimeoutException) as exc:
        return f"Search unavailable right now ({type(exc).__name__}). Say the live search failed and answer cautiously."
    except Exception as exc:  # network errors, parsing issues
        logger.exception("Unexpected search failure")
        return f"Search failed ({type(exc).__name__}). Say the live search failed and answer cautiously."

    if _last_error[0] and "No results found" not in _last_error[0]:
        return (f"Live search failed for '{query}' (error: {_last_error[0]}). "
                "Tell the customer the live search is unavailable right now.")
    return f"No results found for '{query}'."
