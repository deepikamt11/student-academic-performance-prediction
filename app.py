
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained prediction models
with open("student_models.pkl", "rb") as file:
    models = pickle.load(file)

linear_model = models["linear_model"]
forest_model = models["forest_model"]
features = models["features"]


def get_category(mark):
    if mark < 40:
        return "Low Performance"
    elif mark < 70:
        return "Medium Performance"
    return "High Performance"


@app.route("/", methods=["GET", "POST"])
def home():
    # Load the demo student dataset
    df = pd.read_csv("student/academic_performance.csv")

    df["category"] = df["final_marks"].apply(get_category)

    total_students = len(df)
    average_marks = round(df["final_marks"].mean(), 2)
    average_attendance = round(df["attendance"].mean(), 2)

    high_count = int((df["category"] == "High Performance").sum())
    medium_count = int((df["category"] == "Medium Performance").sum())
    low_count = int((df["category"] == "Low Performance").sum())

    # Group marks into ranges for the bar chart
    bins = [0, 40, 50, 60, 70, 80, 101]
    labels = ["0-39", "40-49", "50-59", "60-69", "70-79", "80-100"]
    performance_distribution = (
        pd.cut(
            df["final_marks"],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True
        )
        .value_counts()
        .reindex(labels, fill_value=0)
    )

    prediction = None
    error = None

    if request.method == "POST":
        try:
            student = {
                "attendance": float(request.form["attendance"]),
                "internal_marks": float(request.form["internal_marks"]),
                "assignment_score": float(request.form["assignment_score"]),
                "study_hours": float(request.form["study_hours"]),
                "previous_grade": float(request.form["previous_grade"])
            }

            ranges = {
                "attendance": (0, 100, "Attendance"),
                "internal_marks": (0, 25, "Internal marks"),
                "assignment_score": (0, 20, "Assignment score"),
                "study_hours": (0, 24, "Study hours"),
                "previous_grade": (0, 100, "Previous grade")
            }

            for key, (minimum, maximum, label) in ranges.items():
                value = student[key]
                if not minimum <= value <= maximum:
                    raise ValueError(
                        f"{label} must be between {minimum} and {maximum}."
                    )

            student_df = pd.DataFrame([student], columns=features)

            linear_result = float(linear_model.predict(student_df)[0])
            forest_result = float(forest_model.predict(student_df)[0])

            linear_result = max(0, min(100, linear_result))
            forest_result = max(0, min(100, forest_result))

            prediction = {
                "linear": round(linear_result, 2),
                "linear_category": get_category(linear_result),
                "forest": round(forest_result, 2),
                "forest_category": get_category(forest_result)
            }

        except (ValueError, KeyError) as e:
            error = str(e) or "Please enter valid values."

    dashboard = {
        "total": total_students,
        "average_marks": average_marks,
        "average_attendance": average_attendance,
        "high": high_count,
        "medium": medium_count,
        "low": low_count,
        "performance_labels": labels,
        "performance_values": [
            int(value) for value in performance_distribution.tolist()
        ],
        "category_labels": [
            "High Performance", "Medium Performance", "Low Performance"
        ],
        "category_values": [high_count, medium_count, low_count],
        "attendance_values": [
            int(value) for value in range(50, 101, 10)
        ],
        "marks_by_attendance": [
            round(
                df.loc[
                    (df["attendance"] >= value) &
                    (df["attendance"] < value + 10),
                    "final_marks"
                ].mean(), 2
            ) if not df.loc[
                (df["attendance"] >= value) &
                (df["attendance"] < value + 10)
            ].empty else 0
            for value in range(50, 101, 10)
        ]
    }

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        dashboard=dashboard
    )


if __name__ == "__main__":
    app.run(debug=True)
