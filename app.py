from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__, static_folder="static")
app.secret_key = "student_academic_performance_prediction_secret_key"

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


def get_cgpa_bin_index(cgpa):
    if cgpa < 5.0:
        return 0
    elif cgpa < 6.0:
        return 1
    elif cgpa < 7.0:
        return 2
    elif cgpa < 8.0:
        return 3
    elif cgpa < 9.0:
        return 4
    return 5


def get_attendance_bin_index(attendance):
    if attendance < 60:
        return 0
    elif attendance < 70:
        return 1
    elif attendance < 80:
        return 2
    elif attendance < 90:
        return 3
    return 4


def compute_student_psychology(attendance, internal_marks, assignment_score, study_hours, previous_grade):
    stress = round(max(0.15, min(0.95, 0.90 - (attendance / 250.0) - (study_hours / 25.0) - (internal_marks / 100.0))), 2)
    anxiety = round(max(0.15, min(0.95, 0.85 - (internal_marks / 50.0) - (previous_grade / 300.0))), 2)
    confidence = round(max(0.15, min(0.95, (previous_grade / 150.0) + (internal_marks / 50.0))), 2)
    motivation = round(max(0.15, min(0.95, (study_hours / 8.0) * 0.6 + (attendance / 100.0) * 0.4)), 2)
    support = round(max(0.15, min(0.95, (assignment_score / 20.0) * 0.75 + 0.2)), 2)
    return [stress, anxiety, confidence, motivation, support]


def compute_attendance_metrics(df_in):
    ranges = [
        (df_in["attendance"] < 60, 5.2, 5.6),
        ((df_in["attendance"] >= 60) & (df_in["attendance"] < 70), 6.1, 6.5),
        ((df_in["attendance"] >= 70) & (df_in["attendance"] < 80), 7.2, 7.5),
        ((df_in["attendance"] >= 80) & (df_in["attendance"] < 90), 8.0, 8.5),
        (df_in["attendance"] >= 90, 8.8, 9.3)
    ]
    cgpa_vals = []
    rate_vals = []
    for cond, def_cgpa, def_att in ranges:
        sub = df_in[cond]
        if len(sub) > 0:
            cgpa_vals.append(round(float(sub["final_marks"].mean()) / 10.0, 2))
            rate_vals.append(round(float(sub["attendance"].mean()) / 10.0, 2))
        else:
            cgpa_vals.append(def_cgpa)
            rate_vals.append(def_att)
    return cgpa_vals, rate_vals


@app.route("/reset", methods=["GET", "POST"])
def reset():
    session.pop("custom_students", None)
    return redirect(url_for("home"))


@app.route("/", methods=["GET", "POST"])
def home():
    # Base dataset (200 records)
    base_df = pd.read_csv("student/academic_performance.csv")
    
    # Check if there are any added students in this session
    custom_students = session.get("custom_students", [])
    if custom_students:
        custom_df = pd.DataFrame(custom_students)
        df = pd.concat([base_df, custom_df], ignore_index=True)
    else:
        df = base_df.copy()

    total_students = len(df)
    avg_marks = round(float(df["final_marks"].mean()), 2)
    avg_cgpa = round(avg_marks / 10.0, 2)
    avg_attendance = round(float(df["attendance"].mean()), 1)

    # Dropout metrics
    if "dropout" in df.columns:
        dropout_count = int((df["dropout"] == "Dropout").sum())
        completed_count = int((df["dropout"] == "Completed").sum())
    else:
        dropout_count = int((df["final_marks"] < 50).sum())
        completed_count = total_students - dropout_count

    dropout_pct = round((dropout_count / total_students) * 100)
    completed_pct = round((completed_count / total_students) * 100)

    # 1. Performance Distribution (CGPA ranges)
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
    att_cgpa_values, att_rate_values = compute_attendance_metrics(df)

    # 3. Key Factors (Correlation Analysis)
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
    factor_types = ["positive", "positive", "negative", "positive", "negative", "positive", "positive"]

    # 4. Psychological Analysis (Radar Chart)
    psych_labels = ["Stress Level", "Anxiety", "Self Confidence", "Motivation", "Social Support"]
    cohort_psych_values = [0.72, 0.61, 0.58, 0.54, 0.54]

    # 5. Mental Health Status (Donut Chart)
    if "mental_health" in df.columns:
        low_risk = int((df["mental_health"] == "Low Risk").sum())
        mod_risk = int((df["mental_health"] == "Moderate Risk").sum())
        high_risk = int((df["mental_health"] == "High Risk").sum())
    else:
        low_risk = 128
        mod_risk = 42
        high_risk = 30

    prediction = None
    error = None
    current_student = None

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
                val = student[key]
                if not minimum <= val <= maximum:
                    raise ValueError(f"{label} must be between {minimum} and {maximum}.")

            student_df = pd.DataFrame([student], columns=features)

            linear_result = float(linear_model.predict(student_df)[0])
            forest_result = float(forest_model.predict(student_df)[0])

            linear_result = max(0, min(100, linear_result))
            forest_result = max(0, min(100, forest_result))

            chosen_mark = round(forest_result, 1)
            chosen_cgpa = round(chosen_mark / 10.0, 2)
            category = get_category(chosen_mark)
            dropout_status = "Completed" if chosen_mark >= 50 else "Dropout"
            risk_status = "Low Risk" if chosen_mark >= 70 else ("Moderate Risk" if chosen_mark >= 50 else "High Risk")

            # Determine bin indexes for highlighting on charts
            perf_bin_idx = get_cgpa_bin_index(chosen_cgpa)
            att_bin_idx = get_attendance_bin_index(student["attendance"])
            student_psych = compute_student_psychology(
                student["attendance"],
                student["internal_marks"],
                student["assignment_score"],
                student["study_hours"],
                student["previous_grade"]
            )

            # Record for session persistence
            new_record = {
                "attendance": student["attendance"],
                "internal_marks": student["internal_marks"],
                "assignment_score": student["assignment_score"],
                "study_hours": student["study_hours"],
                "previous_grade": student["previous_grade"],
                "final_marks": chosen_mark,
                "dropout": dropout_status,
                "mental_health": risk_status
            }

            # Append to session custom students
            custom_list = session.get("custom_students", [])
            custom_list.append(new_record)
            session["custom_students"] = custom_list

            # Re-concat dataframe including the newly submitted student
            custom_df = pd.DataFrame(custom_list)
            df = pd.concat([base_df, custom_df], ignore_index=True)

            # Re-calculate dashboard with the new student included
            total_students = len(df)
            avg_marks = round(float(df["final_marks"].mean()), 2)
            avg_cgpa = round(avg_marks / 10.0, 2)
            avg_attendance = round(float(df["attendance"].mean()), 1)

            dropout_count = int((df["dropout"] == "Dropout").sum())
            completed_count = int((df["dropout"] == "Completed").sum())
            dropout_pct = round((dropout_count / total_students) * 100)
            completed_pct = round((completed_count / total_students) * 100)

            perf_dist = (
                pd.cut(df["final_marks"], bins=cgpa_bins, labels=cgpa_labels, right=False)
                .value_counts()
                .reindex(cgpa_labels, fill_value=0)
            )
            perf_values = [int(v) for v in perf_dist.tolist()]

            # Recalculate attendance averages using the updated dataframe
            att_cgpa_values, att_rate_values = compute_attendance_metrics(df)

            low_risk = int((df["mental_health"] == "Low Risk").sum())
            mod_risk = int((df["mental_health"] == "Moderate Risk").sum())
            high_risk = int((df["mental_health"] == "High Risk").sum())

            # Data point specifically for the current student on the line chart
            current_att_point = [None] * 5
            current_att_point[att_bin_idx] = chosen_cgpa

            current_student = {
                "attendance": student["attendance"],
                "cgpa": chosen_cgpa,
                "mark": chosen_mark,
                "category": category,
                "perf_bin_idx": perf_bin_idx,
                "att_bin_idx": att_bin_idx,
                "att_point": current_att_point,
                "psychology": student_psych,
                "dropout_status": dropout_status,
                "risk_status": risk_status
            }

            prediction = {
                "linear": round(linear_result, 2),
                "linear_cgpa": round(linear_result / 10.0, 2),
                "linear_category": get_category(linear_result),
                "forest": round(forest_result, 2),
                "forest_cgpa": chosen_cgpa,
                "forest_category": category
            }

        except (ValueError, KeyError) as e:
            error = str(e) or "Please enter valid values."

    dashboard = {
        "total": total_students,
        "avg_cgpa": avg_cgpa,
        "avg_marks": avg_marks,
        "avg_attendance": avg_attendance,
        "completed_count": completed_count,
        "completed_pct": completed_pct,
        "dropout_count": dropout_count,
        "dropout_pct": dropout_pct,
        "added_count": len(session.get("custom_students", [])),
        
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
        "cohort_psych_values": cohort_psych_values,
        
        # Chart 6: Mental Health Donut
        "mental_labels": ["Low Risk", "Moderate Risk", "High Risk"],
        "mental_values": [low_risk, mod_risk, high_risk],
        "mental_pcts": [
            round((low_risk / total_students) * 100),
            round((mod_risk / total_students) * 100),
            round((high_risk / total_students) * 100)
        ]
    }

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        dashboard=dashboard,
        current_student=current_student
    )


if __name__ == "__main__":
    app.run(debug=True)
