
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained models and evaluation metrics
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
    else:
        return "High Performance"


@app.route("/", methods=["GET", "POST"])
def home():
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

            # Validate input values
            if not 0 <= student["attendance"] <= 100:
                raise ValueError("Attendance must be between 0 and 100.")

            if not 0 <= student["internal_marks"] <= 25:
                raise ValueError("Internal marks must be between 0 and 25.")

            if not 0 <= student["assignment_score"] <= 20:
                raise ValueError("Assignment score must be between 0 and 20.")

            if not 0 <= student["study_hours"] <= 24:
                raise ValueError("Study hours must be between 0 and 24.")

            if not 0 <= student["previous_grade"] <= 100:
                raise ValueError("Previous grade must be between 0 and 100.")

            student_df = pd.DataFrame([student], columns=features)

            linear_result = float(linear_model.predict(student_df)[0])
            forest_result = float(forest_model.predict(student_df)[0])

            # Keep predictions within 0 to 100
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

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
