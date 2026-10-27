# AutoGen Weather News Agent

A small multi-agent app built with **Microsoft AutoGen (AgentChat)**. You type a city; the agent searches the web with **DuckDuckGo** (free, no API key) and replies with a **3-line bullet summary** of the current weather, the short-term forecast, and any alerts.

```
Enter a city (or 'exit' to quit): chennai

Weather in Chennai:
- Current conditions: Clear with a temperature of 29°C.
- Forecast: Sunny with highs around 30°C for the next two days.
- No major alerts (accuweather.com).
```

---

## How it works

```
User city
   │
   ▼
┌──────────────────────── RoundRobinGroupChat ────────────────────────┐
│                                                                     │
│  weather_reporter ──calls──► search_weather_news() ──► DuckDuckGo   │
│        │  (reflect_on_tool_use=True: writes 3 bullets               │
│        │   from the results; up to 2 searches per turn)             │
│        ▼                                                            │
│  weather_reviewer ──► "APPROVE"  or  "fix X" ──► back to reporter   │
│                                                                     │
└──── stops on: APPROVE  |  8 messages  |  90 seconds ────────────────┘
   │
   ▼
Final 3 bullets printed
```

| Component | Role |
|---|---|
| `search_weather_news()` | Async tool with type hints and a docstring; AutoGen turns these into the tool schema the model sees. Tries DuckDuckGo news, then DuckDuckGo web, then other `ddgs` engines, across three query wordings. |
| `weather_reporter` | `AssistantAgent` with `tools=[search_weather_news]`, `reflect_on_tool_use=True` (summarises instead of dumping raw results) and `max_tool_iterations=2` (one retry search if results are weak). Never invents data; says "unavailable" if nothing usable is found. |
| `weather_reviewer` | Checks the format (3 short bullets with a source, or a single "unavailable" line) and replies `APPROVE`. |
| `RoundRobinGroupChat` | Alternates reporter → reviewer until a termination condition fires. |
| Terminations | `TextMentionTermination("APPROVE")` \| `MaxMessageTermination(8)` \| `TimeoutTermination(90)` — whichever comes first, so the team can't loop. |
| `Console` | Streams every message, tool call, tool result and token stats when `VERBOSE = True`. |
| `model_client.close()` | Called in a `finally` block so the HTTP client is always released. |

---

## Setup

**1. Create and activate a virtual environment**

Windows:
```bat
python -m venv autogenv
autogenv\Scripts\activate
```

macOS / Linux:
```bash
python -m venv autogenv
source autogenv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Create a `.env` file** in the same folder as the script:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini   # optional, this is the default
```

**4. Run**
```bash
python weather_news_agent.py
```
Type a city name; type `exit`, `quit` or `q` (or press Enter) to stop.

---

## Settings

At the top of the script:

| Setting | Default | What it does |
|---|---|---|
| `VERBOSE` | `False` | `True` shows the full agent conversation in the Console (tool calls, tool results, reviewer messages, token usage, duration). `False` prints only the final bullets. |
| `SHOW_SEARCH_ERRORS` | `True` | When every search attempt fails, prints a `[search debug]` block with the reason. Set `False` once things are stable. |

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `Weather data unavailable` **with** a `[search debug]` block | All searches failed | Read the debug lines: `Ratelimit`/`202` → wait a minute and retry; `ConnectError`/`timeout`/`SSL` → office proxy, VPN or firewall is blocking the search sites. |
| `Weather data unavailable` **without** a `[search debug]` block | Search worked, but the results had no real weather figures | Retry, or use a larger nearby city. Set `VERBOSE = True` to see exactly what the search returned. |
| Answer appears but numbers look wrong | Check the source in brackets | Run with `VERBOSE = True` and compare the bullets with the `ToolCallExecutionEvent` output. |
| `OPENAI_API_KEY not found` | `.env` missing or in the wrong folder | Put `.env` next to the script. |
| `RuntimeWarning: duckduckgo_search has been renamed to ddgs` | Old package installed | `pip uninstall duckduckgo_search` then `pip install ddgs`. |

Search results vary from run to run (different pages, different fallback engines), so an occasional "unavailable" followed by a successful retry is normal.

---

## Project structure

```
weather_news_autogen/
├── weather_news_agent.py   # the agent app
├── requirements.txt        # Python dependencies
├── README.md               # this file
└── .env                    # your OpenAI key (do not commit)
```

Add `.env` to `.gitignore` if you push this to GitHub.
