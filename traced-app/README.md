# 🔍 LangSmith Tracing Demo (LangChain + OpenAI)

A small LangChain app that sends three prompts to OpenAI `gpt-4o-mini` and prints the answers.
Its purpose is to show that **LangSmith tracing needs no code**: you switch it on and off with
environment variables in `.env`, and the Python file stays the same.

## What it does

Sends these prompts one by one and prints each answer:

1. In one sentence, what does a job scheduler like Control-M do?
2. Give me a PowerShell one-liner that lists the 5 largest files in `C:\Temp`.
3. Explain the difference between latency and throughput in two sentences.

`temperature=0` keeps answers consistent between runs, which makes traces easier to compare.

## Requirements

- Python 3.9+
- An **OpenAI API key**
- A **LangSmith** account and API key (free tier at [smith.langchain.com](https://smith.langchain.com)); only needed for tracing

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

Create a `.env` file next to `traced_app.py`:

```env
OPENAI_API_KEY=sk-...

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=traced-app-demo
```

## Run

```bash
python traced_app.py
```

### Example output

```
--- Prompt 1: In one sentence, what does a job scheduler like Control-M do?
<answer>

--- Prompt 2: Give me a PowerShell one-liner that lists the 5 largest files in C:\Temp.
<answer>

--- Prompt 3: Explain the difference between latency and throughput in two sentences.
<answer>
```

## Turning tracing on and off

| `.env` setting | Result |
|---|---|
| `LANGSMITH_TRACING=true` | Every `llm.invoke()` is sent to LangSmith as a run |
| `LANGSMITH_TRACING=false` (or line removed) | Nothing is traced; the app behaves exactly the same |
| `LANGSMITH_PROJECT=<name>` | The project the runs appear under (created automatically) |

No code change is needed: LangChain reads these variables on its own once `load_dotenv()` has loaded them.

## Viewing the traces

1. Open [smith.langchain.com](https://smith.langchain.com) and go to **Tracing Projects → `traced-app-demo`**.
2. You'll see **three `ChatOpenAI` runs**, one per prompt.
3. Open a run to see the input prompt, the output, the model and its settings, **token usage**, **latency** and cost.

## Troubleshooting

| Problem | Fix |
|---|---|
| No runs in LangSmith | Check `LANGSMITH_TRACING=true` (lowercase `true`) and that `LANGSMITH_API_KEY` is correct |
| Runs in the wrong project | Check `LANGSMITH_PROJECT`; without it, runs go to `default` |
| `401` / authentication error | Wrong `OPENAI_API_KEY` |
| Using the EU region of LangSmith | Add `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com` |
| Older tutorials mention `LANGCHAIN_TRACING_V2` | That's the older variable name; `LANGSMITH_TRACING` is the current one |

## Ideas for next steps

- Add a prompt template plus an output parser chain to see nested runs in a trace
- Tag runs with `config={"tags": [...], "metadata": {...}}` and filter by them in LangSmith
- Save the prompts as a LangSmith **dataset** and run an evaluation

## Project layout

```
.
├── traced_app.py
├── requirements.txt
├── .env              # API keys and tracing settings (do not commit)
└── README.md
```
