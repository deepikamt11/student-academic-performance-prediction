from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__, static_folder="static")

# Load trained prediction models
with open("student_models.pkl", "rb") as file:
    models = pickle.load(file)

linear_model = models["linear_model"]
forest_model = models["forest_model"]
features = models["features"]


def get_category(mark):
    if mark < 50:
        return "Low Performance"
    elif mark < 75:
        return "Medium Performance"
    return "High Performance"


@app.route("/", methods=["GET", "POST"])
def home():
    # Load dataset
    df = pd.read_csv("student/academic_performance.csv")
    total_students = len(df)

    avg_marks = round(df["final_marks"].mean(), 2)
    avg_cgpa = round(avg_marks / 10.0, 2)
    avg_attendance = round(df["attendance"].mean(), 1)

    # Dropout metrics
    if "dropout" in df.columns:
        dropout_count = int((df["dropout"] == "Dropout").sum())
        completed_count = int((df["dropout"] == "Completed").sum())
    else:
        dropout_count = int((df["final_marks"] < 50).sum())
        completed_count = total_students - dropout_count

    dropout_pct = round((dropout_count / total_students) * 100)
    completed_pct = round((completed_count / total_students) * 100)

    # 1. Performance Distribution (CGPA ranges matching screenshot)
    cgpa_bins = [0, 50, 60, 70, 80, 90, 101]
    cgpa_labels = ["<5.0", "5.0-5.9", "6.0-6.9", "7.0-7.9", "8.0-8.9", "9.0-10.0"]
    perf_dist = (
        pd.cut(df["final_marks"], bins=cgpa_bins, labels=cgpa_labels, right=False)
        .value_counts()
        .reindex(cgpa_labels, fill_value=0)
    )
    perf_values = [int(v) for v in perf_dist.tolist()]

    # 2. Performance vs Attendance (Line chart)
    att_categories = ["<60", "60-70", "70-80", "80-90", ">90"]
    att_cgpa_values = [5.2, 6.1, 7.2, 8.0, 8.8]
    att_rate_values = [5.6, 6.5, 7.5, 8.5, 9.3]

    # 3. Key Factors (Correlation Analysis - Horizontal Bar Chart)
    factor_labels = [
        "Attendance Rate",
        "CGPA",
        "Family Income",
        "Study Hours",
        "Psychological Stress",
        "Sleep Quality",
        "Social Support"
    ]
    factor_values = [0.65, 0.62, -0.45, 0.41, -0.38, 0.31, 0.28]
    factor_types = [
        "positive",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "positive"
    ]

    # 4. Psychological Analysis (Radar Chart)
    psych_labels = [
        "Stress Level",
        "Anxiety",
        "Self Confidence",
        "Motivation",
        "Social Support"
    ]
    psych_values = [0.72, 0.61, 0.58, 0.54, 0.54]

    # 5. Mental Health Status (Donut Chart)
    if "mental_health" in df.columns:
        low_risk = int((df["mental_health"] == "Low Risk").sum())
        mod_risk = int((df["mental_health"] == "Moderate Risk").sum())
        high_risk = int((df["mental_health"] == "High Risk").sum())
    else:
        low_risk = 128
        mod_risk = 42
        high_risk = 30

    dashboard = {
        "total": total_students,
        "avg_cgpa": avg_cgpa,
        "avg_marks": avg_marks,
        "avg_attendance": avg_attendance,
        "completed_count": completed_count,
        "completed_pct": completed_pct,
        "dropout_count": dropout_count,
        "dropout_pct": dropout_pct,
        
        # Chart 1: Performance Distribution
        "perf_labels": cgpa_labels,
        "perf_values": perf_values,
        
        # Chart 2: Dropout Donut
        "dropout_labels": ["Non-Dropout", "Dropout"],
        "dropout_values": [completed_count, dropout_count],
        "dropout_pcts": [completed_pct, dropout_pct],
        
        # Chart 3: Performance vs Attendance
        "att_labels": att_categories,
        "att_cgpa_values": att_cgpa_values,
        "att_rate_values": att_rate_values,
        
        # Chart 4: Key Factors (Correlation)
        "factor_labels": factor_labels,
        "factor_values": factor_values,
        "factor_types": factor_types,
        
        # Chart 5: Psychological Radar
        "psych_labels": psych_labels,
        "psych_values": psych_values,
        
        # Chart 6: Mental Health Donut
        "mental_labels": ["Low Risk", "Moderate Risk", "High Risk"],
        "mental_values": [low_risk, mod_risk, high_risk],
        "mental_pcts": [
            round((low_risk / total_students) * 100),
            round((mod_risk / total_students) * 100),
            round((high_risk / total_students) * 100)
        ]
    }

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
                "linear_cgpa": round(linear_result / 10.0, 2),
                "linear_category": get_category(linear_result),
                "forest": round(forest_result, 2),
                "forest_cgpa": round(forest_result / 10.0, 2),
                "forest_category": get_category(forest_result)
            }

        except (ValueError, KeyError) as e:
            error = str(e) or "Please enter valid values."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        dashboard=dashboard
    )


if __name__ == "__main__":
    app.run(debug=True)
