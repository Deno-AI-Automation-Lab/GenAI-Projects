# 🥣 Healthy Breakfast Ideas (LangChain + OpenAI)

A minimal LangChain script that asks OpenAI's `gpt-4o-mini` for **exactly 5 healthy breakfast ideas**
and prints them as a clean numbered list. It shows how to get strict, predictable output formatting
through prompt instructions alone.

## Example output

```
1. Greek yogurt with mixed berries, honey and a sprinkle of chia seeds
2. Overnight oats with almond milk, banana slices and walnuts
3. Vegetable ragi dosa with coconut chutney
4. Whole-grain toast with avocado and a poached egg
5. Moong dal chilla stuffed with spinach and paneer
```

(Ideas vary on each run because `temperature=0.7`.)

## Requirements

- Python 3.9+
- An OpenAI API key

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

Create a `.env` file in the same folder:

```env
OPENAI_API_KEY=sk-...
```

## Run

```bash
python breakfast.py
```

## How it works

1. `load_dotenv()` loads `OPENAI_API_KEY` from `.env`.
2. `ChatOpenAI(model="gpt-4o-mini", temperature=0.7)` creates the chat model.
3. The prompt sets strict format rules: numbered 1–5, one line per idea, no preamble, heading or closing remarks.
4. `llm.invoke(prompt)` sends the request, and `reply.content` is printed.

## Customise

| Change | Where |
|---|---|
| Model | `model="gpt-4o-mini"` |
| Creativity (0 = same every time, 1 = more varied) | `temperature=0.7` |
| Number or type of ideas (e.g. "South Indian", "high-protein", "under 300 kcal") | the `prompt` text |

## Project layout

```
.
├── breakfast.py
├── requirements.txt
├── .env          # your API key (do not commit)
└── README.md
```

Add `.env` and `.venv/` to `.gitignore`.
