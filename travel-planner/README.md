# Travel Planner (CrewAI)

A simple CrewAI script. You enter a city, and a **Travel Planner** agent recommends the top 3 places to visit there. For each place it explains why it's worth visiting and suggests the best time of day to go.

## How it works

| Component | Details |
|-----------|---------|
| **Agent** | Travel Planner — a seasoned travel guide with 15+ years of global travel experience |
| **Task** | List Places — find the top 3 places in the city, one reason each, and the best time of day to visit |
| **Crew** | One agent, one task, run with `crew.kickoff()` |

When the script starts it asks for a city with `input()`. The city is passed as `inputs={"city": city}` and fills the `{city}` placeholder in the agent's goal and the task description.

## Project structure

```
.
├── travel_planner.py   # Main script
├── requirements.txt    # Python dependencies
├── .env                # API key and model (not committed)
└── README.md
```

## Prerequisites

- Python 3.10 – 3.13 (required by CrewAI)
- An OpenAI API key

## Setup

1. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** in the project folder:

   ```
   OPENAI_API_KEY=your_openai_api_key
   MODEL=gpt-4o-mini
   ```

   Add `.env` to `.gitignore` so your key is never committed.

## Run

```bash
python travel_planner.py
```

## Example

```
Enter a city: Chennai

Top 3 places to visit in Chennai:

• Marina Beach – One of the longest urban beaches in the world and the heart of local life – Best time: Evening (cooler air, sunset crowds and street food stalls)
• Kapaleeshwarar Temple – A striking Dravidian-style temple central to Mylapore's culture – Best time: Morning (peaceful early prayers and fewer visitors)
• Fort St. George – India's first British fortress, with a museum of colonial history – Best time: Morning (cooler temperatures before the midday heat)
```

With `verbose=True`, you'll also see the agent's reasoning steps in the console before the final result. The actual output varies by model and run.

## Customisation

- **Change the model:** edit `MODEL` in `.env` (for example `gpt-4o`).
- **Change the number of places:** update "3" in the agent goal, task description and expected output.
- **Quieter output:** set `verbose=False` on the Agent and the Crew.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `AuthenticationError` | Check that `OPENAI_API_KEY` in `.env` is correct and has credit |
| `ModuleNotFoundError: crewai` | Activate the virtual environment and run `pip install -r requirements.txt` |
| Model not found | Make sure `MODEL` in `.env` is a valid OpenAI model name |
| Install fails on Python 3.14+ | Use Python 3.10 – 3.13 |
