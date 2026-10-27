"""
AutoGen Weather News Agent
--------------------------
Takes a city from the user, searches the web with DuckDuckGo (free, no API key),
and replies with a 3-bullet weather summary.

Team (RoundRobinGroupChat):
  weather_reporter -> calls the search tool (up to 2 searches), writes 3 short bullets
                      (reflect_on_tool_use=True)
  weather_reviewer -> checks the format; replies APPROVE or asks for a fix

Terminations (whichever fires first): reviewer says APPROVE | 8 messages | 90 seconds

Output:
  VERBOSE = False -> prints only the final 3 bullets
  VERBOSE = True  -> Console shows every agent message, tool call and token stats

Setup:
  pip install -U autogen-agentchat "autogen-ext[openai]" ddgs python-dotenv
  .env -> OPENAI_API_KEY=sk-...   (optional: OPENAI_MODEL=gpt-4o-mini)
"""

import asyncio
import os
import time
from typing import Annotated

from dotenv import load_dotenv
from ddgs import DDGS

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
    TimeoutTermination,
)
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

VERBOSE = False  # True = full agent conversation via Console; False = final answer only
SHOW_SEARCH_ERRORS = True  # print why a search failed (set False once things are stable)
REPORTER = "weather_reporter"


# ---------------------------------------------------------------------------
# Tool: DuckDuckGo search
# ---------------------------------------------------------------------------
def _ddg_search(city: str, max_results: int) -> tuple[list[dict], list[str]]:
    """Blocking search with fallbacks. Returns (results, errors seen along the way)."""
    ddgs = DDGS()
    queries = [f"{city} weather", f"{city} weather forecast today", f"{city} temperature"]
    attempts = [
        ("news", "duckduckgo"),
        ("text", "duckduckgo"),
        ("text", "auto"),  # lets ddgs rotate to Bing/Brave/etc. if DuckDuckGo is empty
    ]
    errors: list[str] = []
    for query in queries:  # different wording often fixes "No results found"
        for kind, backend in attempts:
            try:
                search = ddgs.news if kind == "news" else ddgs.text
                results = list(search(query, max_results=max_results, backend=backend) or [])
                if results:
                    return results, errors
                errors.append(f"{kind}/{backend} '{query}': empty")
            except Exception as exc:
                errors.append(f"{kind}/{backend} '{query}': {type(exc).__name__}: {exc}")
            time.sleep(1)  # brief pause so DuckDuckGo doesn't rate-limit the next try
    return [], errors


async def search_weather_news(
    city: Annotated[str, "Name of the city to get current weather and weather news for, e.g. 'Chennai'"],
    max_results: Annotated[int, "How many search results to fetch (1-8)"] = 5,
) -> str:
    """Search the web (DuckDuckGo) for the latest weather conditions, forecast and
    weather-related news for a given city.

    Use this tool whenever the user asks about weather, temperature, rain,
    storms, alerts or forecasts for a city. Returns a plain-text list of search
    results (title, date, snippet, source URL) for you to summarise.
    """
    max_results = max(1, min(max_results, 8))
    results, errors = await asyncio.to_thread(_ddg_search, city, max_results)

    if not results:
        if SHOW_SEARCH_ERRORS:  # printed for you, not sent to the model
            print(f"[search debug] all attempts failed for '{city}':")
            for e in errors:
                print("   ", e[:200])
        return f"SEARCH_FAILED: no results found for '{city}'."

    lines = []
    for i, r in enumerate(results, start=1):
        date = f" ({r['date']})" if r.get("date") else ""
        url = r.get("url") or r.get("href", "")
        lines.append(f"{i}. {r.get('title', '')}{date}\n   {r.get('body', '')}\n   Source: {url}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agents + team
# ---------------------------------------------------------------------------
def build_team(model_client: OpenAIChatCompletionClient) -> RoundRobinGroupChat:
    reporter = AssistantAgent(
        name=REPORTER,
        model_client=model_client,
        tools=[search_weather_news],
        reflect_on_tool_use=True,  # write an answer from the results, not a raw dump
        max_tool_iterations=2,  # allows one extra search if the first results are weak
        system_message=(
            "You are a Weather News reporter. For the city in the task, ALWAYS call "
            "search_weather_news first. Then reply with EXACTLY 3 bullet points, "
            "each one short line (max ~20 words):\n"
            "- Current conditions / temperature\n"
            "- Forecast for the next day or two\n"
            "- Any alert or notable weather news (or 'No major alerts')\n"
            "Use ONLY facts found in the search results. NEVER invent temperatures or forecasts. "
            "If the results don't contain actual weather figures, call search_weather_news "
            "once more with a different city spelling or a nearby major city before giving up. "
            "If the tool result starts with SEARCH_FAILED, or the second search also has no "
            "weather figures, reply with exactly one line: "
            "'- Weather data unavailable for <city> right now.' "
            "End the last bullet with the source site in brackets, e.g. (timesofindia.com). "
            "No intro, no sign-off. If the reviewer asks for a fix, rewrite the bullets."
        ),
        description="Searches the web and writes a 3-bullet weather summary.",
    )

    reviewer = AssistantAgent(
        name="weather_reviewer",
        model_client=model_client,
        system_message=(
            "You review the weather_reporter's latest answer. "
            "Approve if EITHER: (a) exactly 3 short bullet points about the requested city, "
            "with a source site named in brackets; OR (b) a single line saying weather data "
            "is unavailable. If it passes, reply with only the word APPROVE. "
            "Otherwise give one short instruction on what to fix."
        ),
        description="Checks the summary format and approves it.",
    )

    termination = (
        TextMentionTermination("APPROVE", sources=["weather_reviewer"])
        | MaxMessageTermination(8)
        | TimeoutTermination(90)
    )

    return RoundRobinGroupChat(participants=[reporter, reviewer], termination_condition=termination)


def final_answer(messages) -> str:
    """Return the reporter's last text reply (the approved bullets)."""
    for msg in reversed(messages):
        if isinstance(msg, TextMessage) and msg.source == REPORTER:
            return msg.content
    return "- No answer produced."


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not found. Add it to your .env file.")

    model_client = OpenAIChatCompletionClient(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    team = build_team(model_client)

    try:
        while True:
            city = input("\nEnter a city (or 'exit' to quit): ").strip()
            if not city or city.lower() in {"exit", "quit", "q"}:
                break

            task = f"Give me the latest weather news for {city}."
            if VERBOSE:
                result = await Console(team.run_stream(task=task), output_stats=True)
            else:
                result = await team.run(task=task)

            print(f"\nWeather in {city.title()}:\n{final_answer(result.messages)}")
            await team.reset()  # fresh conversation for the next city
    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())