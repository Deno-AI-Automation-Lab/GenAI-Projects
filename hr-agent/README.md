# HR Assistant Agent (AutoGen)

A minimal AI agent built with Microsoft **AutoGen** (AgentChat API, v0.4+).
An `AssistantAgent` acting as an HR assistant writes a 5-line, bullet-point
job description for a **Junior Python Developer** and streams the reply to the
terminal.

## How it works

| Piece | What it does |
|-------|--------------|
| `OpenAIChatCompletionClient` | Model client that talks to OpenAI (`gpt-4o-mini` by default) |
| `AssistantAgent` | The agent, with `system_message="You are an HR assistant."` |
| `Console(agent.run_stream(...))` | Streams the task and the agent's reply to the terminal |
| `asyncio.run(main())` | AutoGen is async, so all logic lives in `async def main()` |
| `model_client.close()` | Called in a `finally` block so the script exits cleanly instead of hanging |

The OpenAI API key is read from a `.env` file using `python-dotenv`.

## Project structure

```
.
├── hr_agent.py        # The agent script
├── requirements.txt   # Python dependencies
├── .env.example       # Template for your API key
└── README.md
```

## Prerequisites

- Python 3.10 or later
- An OpenAI API key

## Setup

1. **Create and activate a virtual environment**

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   Linux / macOS:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API key**

   Copy `.env.example` to `.env` and set your key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```
   Keep `.env` out of version control (add it to `.gitignore`).

## Run

```bash
python hr_agent.py
```

## Sample output

```
---------- TextMessage (user) ----------
Write a 5-line job description in bullet points for a Junior Python Developer.
---------- TextMessage (hr_assistant) ----------
- Develop and maintain Python applications under the guidance of senior developers.
- Write clean, testable code and participate in code reviews.
- Work with REST APIs, databases, and version control (Git).
- Debug issues and help improve application performance.
- Bachelor's degree in CS or related field; 0–2 years of Python experience.
```

The exact wording will vary from run to run.

## Customisation

- **Change the model:** edit `model="gpt-4o-mini"` in `hr_agent.py`.
- **Change the role:** edit the `system_message`.
- **Change the request:** edit the `task` string.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OPENAI_API_KEY not found` | Make sure `.env` exists next to `hr_agent.py` and contains the key |
| `ModuleNotFoundError: autogen_agentchat` | Run `pip install -r requirements.txt` inside the activated venv |
| Script hangs after the reply | Make sure `await model_client.close()` is still in the `finally` block |
| `401 Unauthorized` | The API key is invalid or has no credit |
