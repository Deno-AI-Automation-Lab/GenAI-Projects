"""Grading thresholds and mark validation from the original grade_system.py."""
import math


def grade_for(mark: float) -> str:
    if not math.isfinite(mark) or not 0 <= mark <= 100:
        raise ValueError("The mark must be a finite number between 0 and 100.")
    if mark >= 90:
        return "A"
    elif mark >= 80:
        return "B"
    elif mark >= 70:
        return "C"
    elif mark >= 60:
        return "D"
    return "E"


def class_figures(students: list[dict]) -> tuple[float, float, float] | None:
    if not students:
        return None
    marks = [student["Mark"] for student in students]
    return sum(marks) / len(marks), max(marks), min(marks)
