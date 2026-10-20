# 🎓 Student Grade Calculator (Command Line)

A small command-line Python program that asks for a student's mark and prints the letter grade.
It is the original console version of the grading logic used in the Streamlit grading app.

## Grading scale

| Grade | Mark |
|---|---|
| A | ≥ 90 |
| B | ≥ 80 |
| C | ≥ 70 |
| D | ≥ 60 |
| E | < 60 |

## Requirements

- Python 3.6+
- No third-party packages; it uses only the standard library

## Run

```bash
python grade_system.py
```

## Examples

```
Enter Student mark (0 to 100): 85.5
Mark: 85.5 | Grade: B
```

```
Enter Student mark (0 to 100): abc
Error: enter a valid number between 0 and 100.
```

```
Enter Student mark (0 to 100): 105
Error: the mark must be a number between 0 and 100.
```

## Input validation

| Input | Result |
|---|---|
| Whole or decimal number 0–100 (`75`, `89.5`) | Grade shown |
| Text or empty input (`abc`, ``) | "enter a valid number" error |
| Out of range (`-5`, `101`) | "between 0 and 100" error |
| `nan`, `inf` | Rejected via `math.isfinite()` |

Spaces around the input are trimmed. The mark is echoed back exactly as you typed it.

## Project layout

```
.
├── grade_system.py
├── requirements.txt
└── README.md
```

## Ideas for next steps

- Loop to grade several students, then print the class average, highest and lowest marks
- Move the grade rules into a reusable `grade_for()` function (as done in `grading.py`)
- Add `pytest` tests for the boundary marks (59.99, 60, 89.99, 90, 100)
