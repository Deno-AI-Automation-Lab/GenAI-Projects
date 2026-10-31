# Customer Support Chatbot — AutoGen multi-agent

Five department agents (Healthy Food, Travel, Exercise, Movies, Weather), a router that
decides who answers, and a quality observer that watches every answer. Each specialist
searches the web with DuckDuckGo (`ddgs`, free, no key) and replies in 5 short bullets.

## Architecture

```
user ─► router ──ROUTE: x──► specialist ──web_search()──► DuckDuckGo
          │                      │  reflect_on_tool_use=True → 5 bullets
   OUT_OF_SCOPE (stop)           ▼
                          quality_observer ── APPROVED (stop)
                                 └── REVISE ─► same specialist (once more)
```

| Piece | Where | What it does |
|---|---|---|
| Router (main agent) | `agents.py` | LLM reads the question + memory, replies `ROUTE: <agent>` + `QUESTION: <standalone rewrite>` (resolves "this movie"), or `OUT_OF_SCOPE` |
| 5 specialists | `agents.py` | `AssistantAgent(tools=[web_search], reflect_on_tool_use=True, memory=[...])` |
| Quality observer | `agents.py` | Fact checker: an `EvidenceContext` attaches every search/weather result to its context, so it always verifies each cited name/date/number |
| Team | `team.py` | `SelectorGroupChat` with a `selector_func`, so routing after the router costs no extra LLM calls and can't wander |
| Terminations | `team.py` | APPROVED · OUT_OF_SCOPE · revision cap (`MAX_REVISIONS=1`) · `MaxMessageTermination(8)` · `TimeoutTermination(120s)` · `TokenUsageTermination(30k)` |
| Live weather | `weather_tool.py` | `get_weather(location)` — Open-Meteo (free, no key): current temp, feels-like, humidity, wind, rain chance, daily forecast. Weather agent only |
| Evidence | `search_tool.py` | Every result gets an id [S1]..; specialists cite one per bullet; citations become a Sources line |
| Search tool | `search_tool.py` | Typed, docstring'd async function. `news=True` + `timelimit` (d/w/m/y) for current info; widens automatically if empty (drops date words, then falls back from DuckDuckGo to other ddgs engines); dated results newest first |
| Memory | `memory_store.py` | Custom AutoGen `Memory`: injects saved facts + last 6 turns into router and specialists; every turn saved to `data/memory.json` |
| Trace observer | `observability.py` | Live console view of every hop + JSONL audit log in `logs/` (speaker, tool calls, tokens) |

## Run

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
copy .env.example .env                               # then put your OPENAI_API_KEY in .env
python main.py                                       # chat with live agent trace
python main.py --quiet                               # answers only
python main.py --ask "Healthy breakfast ideas for diabetics?"
```

Chat commands: `/remember I live in Chennai`, `/memory` (shows file path), `/history`, `/forget`, `/exit`.

## Tests (offline, no key or network)

```bash
python -m pytest -q tests
```

Covers: route → search → revise → approve, out-of-scope, the loop guard stopping endless
revisions, and the keyword fallback router.

## Notes

- The model client is closed in a `finally` block in `main.py`, even on errors.
- If the router's reply can't be parsed, or it says OUT_OF_SCOPE while the question clearly matches a department, a keyword classifier picks the department.
- Today's date is added to every question, so "now" is always correct even if the app runs for days.
- Each question starts a fresh team chat (`team.reset()`); follow-ups work through memory.
- DuckDuckGo rate-limits heavy use; the tool backs off and retries 3 times, then tells the agent to say the search failed.
