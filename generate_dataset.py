
import pandas as pd
import numpy as np
import os

# Reproducible sample data
np.random.seed(42)

# Number of sample students
n = 300

# Generate student input details
attendance = np.random.randint(50, 101, n)
internal_marks = np.random.randint(0, 26, n)
assignment_score = np.random.randint(0, 21, n)
study_hours = np.round(np.random.uniform(0, 8, n), 1)
previous_grade = np.random.randint(0, 101, n)

# Generate sample final marks using a formula
noise = np.random.normal(0, 5, n)

final_marks = (
    0.20 * attendance
    + 1.00 * internal_marks
    + 0.75 * assignment_score
    + 2.00 * study_hours
    + 0.30 * previous_grade
    - 10
    + noise
)

# Keep marks within 0 to 100
final_marks = np.clip(final_marks, 0, 100)
final_marks = np.round(final_marks, 1)

# Create dataset
data = pd.DataFrame({
    "attendance": attendance,
    "internal_marks": internal_marks,
    "assignment_score": assignment_score,
    "study_hours": study_hours,
    "previous_grade": previous_grade,
    "final_marks": final_marks
})

# Create student folder if it does not exist
os.makedirs("student", exist_ok=True)

# Save dataset
data.to_csv("student/academic_performance.csv", index=False)

print("Sample dataset created successfully!")
print("Total students:", len(data))
print("\nFirst 5 records:")
print(data.head())

print("\nDataset saved at:")
print("student/academic_performance.csv")
