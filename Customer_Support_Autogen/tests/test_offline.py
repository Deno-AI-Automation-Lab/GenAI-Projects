"""Offline end-to-end tests: scripted model + fake search, no API key or network needed.

Run:  python -m pytest -q tests   (from the support_bot folder)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from autogen_core import FunctionCall
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import CreateResult, ModelFamily, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient

import search_tool
from config import Settings
from memory_store import ConversationMemory
from observability import TraceObserver
from team import SupportDesk, clean_bullets, keyword_route
import pytest


@pytest.fixture(autouse=True)
def _fresh_evidence():
    search_tool.EVIDENCE.reset()
    yield

MODEL_INFO = {"vision": False, "function_calling": True, "json_output": False,
              "family": ModelFamily.GPT_4O, "structured_output": False}
USAGE = RequestUsage(prompt_tokens=10, completion_tokens=5)
FIVE = "\n".join(f"- point {i}" for i in range(1, 6))


def evidence_call() -> CreateResult:
    return CreateResult(finish_reason="function_calls", usage=USAGE, cached=False,
                        content=[FunctionCall(id="e1", name="get_search_evidence", arguments="{}")])


def tool_call(query: str, news: bool = False) -> CreateResult:
    args = json.dumps({"query": query, "news": news, "timelimit": "w" if news else "any"})
    return CreateResult(finish_reason="function_calls", usage=USAGE, cached=False,
                        content=[FunctionCall(id="c1", name="web_search", arguments=args)])


def make_desk(tmp_path: Path, script: list):
    client = ReplayChatCompletionClient(script, model_info=MODEL_INFO)
    s = Settings(memory_file=tmp_path / "mem.json", log_dir=tmp_path / "logs")
    memory = ConversationMemory(s.memory_file)
    return SupportDesk(client, memory, s, TraceObserver(s.log_dir, verbose=False)), memory, client


CALLS: list[tuple] = []


def fake_search(kind: str, query: str, max_results: int, timelimit: str | None) -> list[dict]:
    CALLS.append((kind, query, timelimit))
    return [{"title": f"Result for {query}", "body": "Sunny, 31°C.", "url": "https://example.com",
             "date": "2026-10-30T08:00:00+00:00", "source": "Example News"}]


def run(coro):
    return asyncio.run(coro)


def test_route_revise_approve_and_memory_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(search_tool, "_search_sync", fake_search)
    script = ["ROUTE: weather_agent", tool_call("Chennai weather today", news=True),
              "- only\n- three\n- bullets", "REVISE: give exactly 5 bullets", FIVE, "APPROVED"]

    async def go():
        desk, memory, client = make_desk(tmp_path, script)
        ans = await desk.ask("What's the weather in Chennai today?")
        await client.close()
        return ans

    ans = run(go())
    assert ans.agent == "weather_agent" and ans.approved and ans.revisions == 1 and ans.tool_calls == 1
    assert ans.text.count("\n") == 4
    saved = json.loads((tmp_path / "mem.json").read_text())
    assert saved["turns"][0]["question"] == "What's the weather in Chennai today?"
    assert list((tmp_path / "logs").glob("trace-*.jsonl"))


def test_router_wrongly_says_out_of_scope_is_overridden(tmp_path, monkeypatch):
    """Exact failure from the user's session: a movie question rejected by the router."""
    monkeypatch.setattr(search_tool, "_search_sync", fake_search)
    script = ["OUT_OF_SCOPE: We only cover food, travel, exercise, movies and weather.",
              tool_call("Tamil movies running in theatres October 2026", news=True), FIVE, "APPROVED"]

    async def go():
        desk, _, client = make_desk(tmp_path, script)
        ans = await desk.ask("which top tamil movie is running in theatres now")
        await client.close()
        return ans

    ans = run(go())
    assert ans.agent == "movies_agent" and ans.approved


def test_true_out_of_scope_is_saved_to_memory(tmp_path):
    async def go():
        desk, memory, client = make_desk(tmp_path, ["OUT_OF_SCOPE: We cover food, travel, exercise, movies and weather."])
        ans = await desk.ask("Fix my laptop")
        await client.close()
        return ans, memory

    ans, memory = run(go())
    assert ans.agent == "router" and "We cover" in ans.text
    assert len(memory.turns) == 1  # previously out-of-scope turns were dropped


def test_memory_persists_and_feeds_followups(tmp_path):
    async def go():
        m1 = ConversationMemory(tmp_path / "m.json")
        await m1.remember_fact("I live in Chennai")
        await m1.remember_turn("give me details on jailer 2 movie", "movies_agent", "- Jailer 2 stars Rajinikanth")
        m2 = ConversationMemory(tmp_path / "m.json")  # simulate restart
        ctx = UnboundedChatCompletionContext()
        result = await m2.update_context(ctx)
        return m2, (await ctx.get_messages())[0].content, result

    m2, injected, result = run(go())
    assert m2.facts == ["I live in Chennai"] and len(m2.turns) == 1
    assert "I live in Chennai" in injected and "jailer 2" in injected and "Rajinikanth" in injected
    assert len(result.memories.results) == 2


def test_old_memory_file_is_migrated(tmp_path):
    old = [{"content": "User fact: vegetarian", "metadata": {"kind": "fact"}},
           {"content": "Earlier the user asked: best yoga?\nexercise_agent answered:\n- Surya namaskar",
            "metadata": {"agent": "exercise_agent", "time": "2026-10-28T10:00:00"}}]
    (tmp_path / "m.json").write_text(json.dumps(old))
    m = ConversationMemory(tmp_path / "m.json")
    assert m.facts == ["vegetarian"]
    assert m.turns[0]["question"] == "best yoga?" and m.turns[0]["agent"] == "exercise_agent"


def test_search_uses_news_and_widens_when_empty(monkeypatch):
    calls = []

    def news_empty(kind, query, max_results, timelimit):
        calls.append((kind, timelimit))
        return [] if kind == "news" else [{"title": "T", "body": "B", "href": "https://x"}]

    monkeypatch.setattr(search_tool, "_search_sync", news_empty)
    out = run(search_tool.web_search("tamil movies now", news=True, timelimit="w"))
    assert calls == [("news", "w"), ("news", "m"), ("text", "w")]
    assert "widened" in out and "https://x" in out


def test_news_results_show_dates_newest_first(monkeypatch):
    def news(kind, query, max_results, timelimit):
        return [{"title": "Old", "body": "", "url": "u1", "date": "2026-01-01T00:00:00"},
                {"title": "New", "body": "", "url": "u2", "date": "2026-10-30T00:00:00"}]

    monkeypatch.setattr(search_tool, "_search_sync", news)
    out = run(search_tool.web_search("x", news=True))
    assert out.index("New") < out.index("Old") and "Published: 2026-10-30" in out


def test_observer_cannot_loop_on_revisions(tmp_path, monkeypatch):
    """Exact failure from the user's session: observer kept saying REVISE until 8 messages."""
    monkeypatch.setattr(search_tool, "_search_sync", fake_search)
    script = ["ROUTE: movies_agent", tool_call("Tamil box office collection", news=True), FIVE,
              "REVISE: give current info", tool_call("Tamil movie release this Friday", news=True), FIVE,
              "REVISE: still not current"] + ["- should never be used"] * 5

    async def go():
        desk, _, client = make_desk(tmp_path, script)
        ans = await desk.ask("what is the latest trending movie in tamil running in theatres now")
        await client.close()
        return ans

    ans = run(go())
    assert ans.agent == "movies_agent" and ans.text == FIVE
    assert ans.revisions == 2 and ans.status.startswith("best effort") and ans.tool_calls == 2


def test_simplify_query_drops_date_words():
    assert search_tool.simplify_query("Tamil movies running in theatres October 2026") == "Tamil movies in theatres"
    assert search_tool.simplify_query("latest trending Tamil movies now") == "Tamil movies"


def test_search_retries_with_simplified_query(monkeypatch):
    seen = []

    def only_simple_works(kind, query, max_results, timelimit):
        seen.append(query)
        return [{"title": "Box office", "body": "", "url": "u"}] if "2026" not in query else []

    monkeypatch.setattr(search_tool, "_search_sync", only_simple_works)
    out = run(search_tool.web_search("Tamil movies October 2026", news=True, timelimit="w"))
    assert "Box office" in out and seen[-1] == "Tamil movies"


def test_keyword_fallback():
    assert keyword_route("best protein breakfast") == "food_agent"
    assert keyword_route("is it going to rain") == "weather_agent"
    assert keyword_route("which top tamil movie is running in theatres now") == "movies_agent"
    assert keyword_route("fix my laptop") is None


def test_engine_chain_falls_back_from_duckduckgo_to_bing(monkeypatch):
    """User's network: DuckDuckGo News always fails, Bing News works."""
    from ddgs.exceptions import DDGSException
    used = []

    class FakeDDGS:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def news(self, query, backend, **kw):
            used.append(backend)
            if backend == "duckduckgo":
                raise DDGSException("No results found.")
            return [{"title": "Jailer 2 Box Office Collection Day 16", "body": "", "url": "u",
                     "date": "2026-10-30T00:00:00", "source": "Example"}]

        text = news

    monkeypatch.setattr(search_tool, "DDGS", FakeDDGS)
    out = run(search_tool.web_search("Tamil box office collection", news=True, timelimit="w"))
    assert used == ["duckduckgo", "bing"]
    assert "Jailer 2 Box Office" in out and "widened" not in out


def test_followup_is_resolved_fact_checked_and_cited(tmp_path, monkeypatch):
    """User's session: 'in this movie who is the director' after a Dharman question."""
    def dharman_news(kind, query, max_results, timelimit):
        return [{"title": "Nayanthara joins Rajinikanth's Dharman", "body": "Director X confirmed ...",
                 "url": "https://news.example/dharman", "date": "2026-10-29T00:00:00", "source": "Example"}]

    monkeypatch.setattr(search_tool, "_search_sync", dharman_news)
    answer = "\n".join(["- Dharman is directed by Director X. [S1]"] +
                       [f"- detail {i} [S1]" for i in range(2, 5)] + ["- Release date not found. [none]"])
    script = [
        "ROUTE: movies_agent\nQUESTION: Who is the director of Rajinikanth's upcoming film Dharman?",
        tool_call("Dharman director", news=True),
        answer,
        "APPROVED",        # observer gets the evidence injected automatically
    ]

    async def go():
        desk, memory, client = make_desk(tmp_path, script)
        await memory.remember_turn("what is the next upcoming movie for rajinikanth", "movies_agent",
                                   "- Rajinikanth's next film is Dharman.")
        ans = await desk.ask("in this movie who is the director")
        await client.close()
        return ans, memory

    ans, memory = run(go())
    assert ans.approved and ans.tool_calls == 1
    assert ans.resolved_question == "Who is the director of Rajinikanth's upcoming film Dharman?"
    assert "[S1]" not in ans.text and "[none]" not in ans.text
    assert ans.sources == ["https://news.example/dharman"]
    assert memory.turns[-1]["resolved"].endswith("Dharman?")


def test_observer_tool_returns_all_evidence(monkeypatch):
    monkeypatch.setattr(search_tool, "_search_sync", fake_search)
    run(search_tool.web_search("a"))
    out = search_tool.get_search_evidence()
    assert "\n[S1]" in out and "example.com" in out


def test_memory_marks_most_recent_exchange(tmp_path):
    async def go():
        m = ConversationMemory(tmp_path / "m.json")
        await m.remember_turn("jailer 2 details", "movies_agent", "- a")
        await m.remember_turn("next rajini movie", "movies_agent", "- Dharman", resolved="What is Rajinikanth's next film?")
        return m._render()

    text = run(go())
    assert text.index("MOST RECENT") < text.index("What is Rajinikanth's next film?")
    assert text.index("jailer 2 details") < text.index("MOST RECENT")
    assert "never reuse their facts" in text


def test_clean_bullets_strips_citations():
    text, cited = clean_bullets("- A [S1]\n- B [S2, S3]\n- C [none]")
    assert text == "- A\n- B\n- C" and cited == ["S1", "S2", "S3"]


def test_citation_ranges_are_stripped():
    text, cited = clean_bullets("- Not found [S1-S3]\n- B [S4, S6–S7]")
    assert text == "- Not found\n- B" and cited == ["S1", "S2", "S3", "S4", "S6", "S7"]


def test_evidence_lists_searches_made(monkeypatch):
    monkeypatch.setattr(search_tool, "_search_sync", fake_search)
    run(search_tool.web_search("Rajinikanth next film", news=True, timelimit="m"))
    out = search_tool.get_search_evidence()
    assert out.startswith("Searches made: 1") and "Rajinikanth next film" in out


# ---------------- weather ----------------
import weather_tool  # noqa: E402

GEO = {"results": [
    {"name": "Kanyakumari", "admin1": "Tamil Nadu", "country": "India", "country_code": "IN",
     "latitude": 8.08, "longitude": 77.54},
]}
FORECAST = {
    "current": {"time": "2026-10-31T12:30", "temperature_2m": 29.4, "apparent_temperature": 33.1,
                "relative_humidity_2m": 74, "precipitation": 0.0, "weather_code": 2, "wind_speed_10m": 14.2},
    "current_units": {"temperature_2m": "°C", "apparent_temperature": "°C", "wind_speed_10m": "km/h"},
    "daily": {"time": ["2026-10-31", "2026-11-01"], "weather_code": [61, 80],
              "temperature_2m_max": [31.0, 30.2], "temperature_2m_min": [24.8, 24.5],
              "precipitation_probability_max": [55, 70], "precipitation_sum": [2.1, 6.4]},
}


def fake_http(url, params):
    return GEO if "geocoding" in url else FORECAST


def test_get_weather_returns_live_numbers_as_evidence(monkeypatch):
    monkeypatch.setattr(weather_tool, "_http_get_json", fake_http)
    out = run(weather_tool.get_weather("Kanyakumari"))
    assert out.startswith("[S1] Live weather for Kanyakumari, Tamil Nadu, India")
    assert "29.4°C" in out and "feels like 33.1°C" in out and "partly cloudy" in out and "rain chance 70%" in out
    ev = search_tool.get_search_evidence()
    assert "get_weather('Kanyakumari'" in ev and "29.4°C" in ev


def test_get_weather_handles_unknown_place_and_network_error(monkeypatch):
    monkeypatch.setattr(weather_tool, "_http_get_json", lambda u, p: {"results": []})
    assert "no place called" in run(weather_tool.get_weather("Xyzzyville"))

    def boom(u, p):
        raise OSError("network down")

    monkeypatch.setattr(weather_tool, "_http_get_json", boom)
    assert "Live weather lookup failed" in run(weather_tool.get_weather("Chennai"))


def test_weather_question_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(weather_tool, "_http_get_json", fake_http)
    weather_call = CreateResult(finish_reason="function_calls", usage=USAGE, cached=False,
                                content=[FunctionCall(id="w1", name="get_weather",
                                                      arguments='{"location": "Kanyakumari"}')])
    answer = "\n".join(["- It is 29.4°C and partly cloudy in Kanyakumari now. [S1]"] +
                       [f"- detail {i} [S1]" for i in range(2, 6)])
    script = ["ROUTE: weather_agent\nQUESTION: what is the weather now in kanyakumari",
              weather_call, answer, "APPROVED"]

    async def go():
        desk, _, client = make_desk(tmp_path, script)
        ans = await desk.ask("what is the weather now in kanyakumari")
        await client.close()
        return ans

    ans = run(go())
    assert ans.agent == "weather_agent" and ans.approved and ans.tool_calls == 1
    assert "29.4°C" in ans.text and ans.sources and "open-meteo.com" in ans.sources[0]


def test_observer_context_always_contains_evidence(monkeypatch):
    from agents import EvidenceContext
    from autogen_core.models import UserMessage
    monkeypatch.setattr(weather_tool, "_http_get_json", fake_http)
    run(weather_tool.get_weather("Kanyakumari"))

    async def go():
        ctx = EvidenceContext()
        await ctx.add_message(UserMessage(content="check this", source="user"))
        return await ctx.get_messages()

    msgs = run(go())
    assert msgs[0].content == "check this"
    assert msgs[-1].content.startswith("EVIDENCE") and "29.4°C" in msgs[-1].content
