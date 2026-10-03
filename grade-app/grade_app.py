"""Run with: python -m streamlit run app.py"""
import streamlit as st
from grading import class_figures, grade_for

st.set_page_config(page_title="Student Grading", page_icon="🎓", layout="wide")
st.title("Student Grading")
st.caption("Add students, review their grades, and see class results.")

if "students" not in st.session_state:
    st.session_state.students = []

with st.form("add_student", clear_on_submit=True):
    st.subheader("Add a student")
    name = st.text_input("Student name", max_chars=120)
    mark = st.number_input(
        "Mark", min_value=0.0, max_value=100.0,
        value=0.0, step=0.5, format="%.2f",
        help="Enter a mark from 0 to 100. Decimal marks are allowed.",
    )
    submitted = st.form_submit_button("Add student", type="primary")

if submitted:
    if not name.strip():
        st.error("Enter a student name.")
    else:
        try:
            grade = grade_for(mark)
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state.students.append(
                {"Name": name.strip(), "Mark": mark, "Grade": grade}
            )
            st.success(f"Added {name.strip()} — Grade {grade}.")

students = st.session_state.students
figures = class_figures(students)

st.subheader("Class figures")
average, highest, lowest = st.columns(3)
if figures is None:
    average.metric("Class average", "—")
    highest.metric("Highest mark", "—")
    lowest.metric("Lowest mark", "—")
else:
    average.metric("Class average", f"{figures[0]:.2f}")
    highest.metric("Highest mark", f"{figures[1]:.2f}")
    lowest.metric("Lowest mark", f"{figures[2]:.2f}")

st.subheader(f"Student results ({len(students)})")
if students:
    st.dataframe(
        students, hide_index=True, width="stretch",
        column_order=["Name", "Mark", "Grade"],
        column_config={"Mark": st.column_config.NumberColumn("Mark", format="%.2f")},
    )
else:
    st.info("Add your first student to see the results table and class figures.")

st.caption("Grades: A ≥ 90 · B ≥ 80 · C ≥ 70 · D ≥ 60 · E < 60")
st.caption("Results stay available during this browser session. They are not saved to disk.")