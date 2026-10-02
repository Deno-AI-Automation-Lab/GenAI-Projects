"""Ask for a mark and display its letter grade."""

import math


def main():
    entered = input("Enter Student mark (0 to 100): ").strip()

    try:
        mark = float(entered)
    except ValueError:
        print("Error: enter a valid number between 0 and 100.")
        return

    if not math.isfinite(mark) or not 0 <= mark <= 100:
        print("Error: the mark must be a number between 0 and 100.")
        return

    if mark >= 90:
        grade = "A"
    elif mark >= 80:
        grade = "B"
    elif mark >= 70:
        grade = "C"
    elif mark >= 60:
        grade = "D"
    else:
        grade = "E"

    print(f"Mark: {entered} | Grade: {grade}")


if __name__ == "__main__":
    main()