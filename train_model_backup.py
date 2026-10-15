
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the real dataset
df = pd.read_csv("student/student-mat.csv", sep=";")

# Select input features and target
X = df[["studytime", "absences", "G1", "G2"]]
y = df["G3"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nModel:", name)
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 2))

# Predict marks for one example student
sample = pd.DataFrame({
    "studytime": [3],
    "absences": [2],
    "G1": [12],
    "G2": [14]
})

print("\nExample Student Prediction:")
for name, model in models.items():
    predicted_marks = model.predict(sample)[0]
    print(name, "predicted final grade:", round(predicted_marks, 2))
