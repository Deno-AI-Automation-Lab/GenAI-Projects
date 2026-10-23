# Tech News Crew

A two-agent [CrewAI](https://www.crewai.com/) crew that finds today's top 3 technology news stories and turns them into a crisp, 5-line newsletter.

## How it works

| Agent | Role | Tools |
|---|---|---|
| **Researcher** | Senior Tech News Researcher. Searches the web for the most important tech stories of the last 24–48 hours and verifies them against credible sources. | `SerperDevTool` (Google search via Serper) |
| **Writer** | Tech Newsletter Writer. Turns the research into an exactly 5-line newsletter using only facts from the research. | None. It cannot search. |

The tasks run **sequentially** (`Process.sequential`):

1. **Research task.** The Researcher returns a numbered list of 3 stories, each with a headline, summary, why it matters, and source name, URL and date.
2. **Writing task.** It receives the research output through `context=[research_task]`, then writes the newsletter:
   - Line 1: Title with today's date
   - Lines 2–4: One story per line, in one sentence each
   - Line 5: Closing takeaway

### 5-line guardrail

The writing task has a guardrail, `check_five_lines`, that counts the lines in the Writer's answer. If there aren't exactly 5, CrewAI sends the answer back to the Writer with the error and asks for a fix. It retries up to 3 times (`guardrail_max_retries=3`).

## Prerequisites

- Python 3.10 – 3.13
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Serper API key](https://serper.dev/). The free tier is enough for testing.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv crewenv
crewenv\Scripts\activate          # Windows
# source crewenv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the same folder as `news_crew.py`:

```env
OPENAI_API_KEY=your-openai-key
SERPER_API_KEY=your-serper-key

# Optional: defaults to gpt-4o-mini
OPENAI_MODEL_NAME=gpt-4o-mini
```

The script reads both keys from environment variables. If either one is missing, it exits with a message that names the missing key.

## Run

```bash
python news_crew.py
```

### Sample output

```
============================================================
TECH NEWSLETTER
============================================================
Tech Briefing: 23 October 2026
A critical vulnerability in PHP has caused widespread website disruptions, prompting urgent patches to safeguard data security.
Tesla has slashed its 2026 capital expenditure forecast by $5 billion, raising concerns about its future investments in AI and robotics.
Nvidia's CEO reports soaring demand for neocloud services, but limited working capital is stalling growth in this promising sector.
These developments highlight a pivotal moment in tech, where security, investment strategies, and cloud innovation are all under scrutiny.
```

Stories change on every run, because the Researcher searches fresh each time.

## Configuration

| Setting | Where | Notes |
|---|---|---|
| Logging detail | `verbose=` on each Agent and the Crew | `True` shows every search, tool result and agent answer. `False` prints only the final newsletter. |
| Model | `OPENAI_MODEL_NAME` in `.env` | Default is `gpt-4o-mini`. |
| Creativity | `temperature=0.3` in `LLM(...)` | Lower values give more consistent output. |
| Retry limit | `guardrail_max_retries=3` | Older crewai versions call this `max_retries`. |

## Notes

- At the end of a run, CrewAI may ask *"Share this execution trace with CrewAI? [y/N]"*. Pressing Enter, or letting it time out after 20 seconds, means **No**.
- Each run uses a few Serper search credits and OpenAI tokens.
- To lock exact package versions after a successful install, run `pip freeze > requirements.txt`.

## Project structure

```
Tech_news/
├── news_crew.py        # The crew: agents, tasks, guardrail, kickoff
├── requirements.txt    # Python dependencies
├── .env                # API keys (do not commit)
└── README.md
```
