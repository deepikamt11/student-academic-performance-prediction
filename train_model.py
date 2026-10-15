
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the new sample dataset
data = pd.read_csv("student/academic_performance.csv")

# Input features
features = [
    "attendance",
    "internal_marks",
    "assignment_score",
    "study_hours",
    "previous_grade"
]

X = data[features]

# Target: final marks
y = data["final_marks"]

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

# Train Random Forest Regression
forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)

forest_predictions = forest_model.predict(X_test)

# Evaluate both models
print("\n--- Linear Regression ---")
print("MAE:", round(mean_absolute_error(y_test, linear_predictions), 2))
print("R2 Score:", round(r2_score(y_test, linear_predictions), 2))

print("\n--- Random Forest Regression ---")
print("MAE:", round(mean_absolute_error(y_test, forest_predictions), 2))
print("R2 Score:", round(r2_score(y_test, forest_predictions), 2))

# Save models and evaluation results
with open("student_models.pkl", "wb") as file:
    pickle.dump({
        "linear_model": linear_model,
        "forest_model": forest_model,
        "features": features,
        "linear_mae": mean_absolute_error(y_test, linear_predictions),
        "linear_r2": r2_score(y_test, linear_predictions),
        "forest_mae": mean_absolute_error(y_test, forest_predictions),
        "forest_r2": r2_score(y_test, forest_predictions)
    }, file)

print("\nBoth models trained and saved successfully!")
