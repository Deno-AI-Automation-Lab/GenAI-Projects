# 🎓 Student Grading (Streamlit)

A simple Streamlit app for adding students with their marks. It assigns each student a letter grade
and shows the class average, highest mark and lowest mark.

## Features

- Add a student with a name and a mark from 0 to 100 (decimals allowed)
- Automatic letter grade
- Class figures: **average**, **highest** and **lowest** mark
- Results table that updates as you add students
- Input validation (empty names and out-of-range marks are rejected)

## Grading scale

| Grade | Mark |
|---|---|
| A | ≥ 90 |
| B | ≥ 80 |
| C | ≥ 70 |
| D | ≥ 60 |
| E | < 60 |

## Project layout

```
.
├── grade_app.py      # Streamlit UI
├── grading.py        # grading logic (imported by the app)
├── requirements.txt
└── README.md
```

### `grading.py` — the logic, kept separate from the UI

| Function | What it does |
|---|---|
| `grade_for(mark: float) -> str` | Returns `"A"`–`"E"`. Raises `ValueError` if the mark is not a finite number between 0 and 100 (this also rejects `NaN` and `inf`) |
| `class_figures(students: list[dict])` | Returns `(average, highest, lowest)` from each student's `"Mark"`, or `None` when the list is empty |

Because it has no Streamlit dependency, `grading.py` can be reused or tested on its own:

```python
from grading import grade_for, class_figures

grade_for(85.5)        # 'B'
class_figures([{"Mark": 90}, {"Mark": 70}])   # (80.0, 90, 70)
```

## Requirements

- Python **3.10+** (`grading.py` uses the `X | None` type-hint syntax)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

## Run

```bash
python -m streamlit run grade_app.py
```

The app opens at **http://localhost:8501**.

## How to use

1. Enter the student's name and mark, then click **Add student**.
2. The grade appears in the success message and in the results table.
3. The class figures update after every addition.

## Note

Data lives in Streamlit's session state. It stays while the browser tab is open, but it is
**not saved to disk** and is lost on refresh or restart.

## Ideas for next steps

- Save and load students as CSV (`st.download_button` / `st.file_uploader`)
- Edit or delete a student
- Add a grade distribution chart
- Unit tests for `grading.py` with `pytest`, covering the boundaries (59.99, 60, 89.99, 90, 100) and invalid input (-1, 100.5, `float("nan")`)
