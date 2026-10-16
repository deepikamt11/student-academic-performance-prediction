import pandas as pd
import numpy as np
import os

# Reproducible sample data
np.random.seed(42)

# Bin definitions matching the Student Performance Dashboard
bin_defs = [
    ('<5.0', 8, 41.0, 47.0, 44.0),
    ('5.0-5.9', 22, 51.0, 56.0, 53.5),
    ('6.0-6.9', 38, 61.0, 66.0, 63.5),
    ('7.0-7.9', 54, 71.0, 76.0, 73.5),
    ('8.0-8.9', 48, 81.0, 85.0, 83.0),
    ('9.0-10.0', 30, 90.5, 92.5, 91.5),
]

all_marks = []
for label, count, low, high, mean_val in bin_defs:
    vals = np.linspace(low, high, count)
    vals = vals - np.mean(vals) + mean_val
    all_marks.extend(vals)

all_marks = np.array(all_marks)
all_marks = np.round(all_marks, 1)

# Shuffle with fixed random state
perm = np.random.RandomState(42).permutation(len(all_marks))
all_marks = all_marks[perm]

# Correlated features
attendance_base = 55 + (all_marks / 100.0) * 35 + np.random.normal(0, 3, 200)
attendance = np.clip(attendance_base, 50, 100)
attendance = attendance - np.mean(attendance) + 82.5
attendance = np.clip(attendance, 55, 100)
attendance = np.round(attendance, 1)

internal_marks = np.clip(np.round(5 + (all_marks / 100.0) * 19 + np.random.normal(0, 1.2, 200), 1), 5, 25)
assignment_score = np.clip(np.round(4 + (all_marks / 100.0) * 15 + np.random.normal(0, 1.0, 200), 1), 4, 20)
study_hours = np.clip(np.round(1.5 + (all_marks / 100.0) * 5.5 + np.random.normal(0, 0.5, 200), 1), 1.0, 8.0)
previous_grade = np.clip(np.round(40 + (all_marks / 100.0) * 55 + np.random.normal(0, 3.5, 200), 1), 35, 100)

# Dropout status: 40 Dropped Out (20%), 160 Completed (80%)
risk_score = (100 - all_marks) * 0.6 + (100 - attendance) * 0.4
dropout_idx = np.argsort(risk_score)[-40:]
dropout = np.array(['Completed'] * 200)
dropout[dropout_idx] = 'Dropout'

# Mental health / Risk: 128 Low Risk (64%), 42 Moderate Risk (21%), 30 High Risk (15%)
high_risk_idx = np.argsort(risk_score)[-30:]
mod_risk_idx = np.argsort(risk_score)[-72:-30]
mental_health = np.array(['Low Risk'] * 200)
mental_health[mod_risk_idx] = 'Moderate Risk'
mental_health[high_risk_idx] = 'High Risk'

# Create DataFrame
data = pd.DataFrame({
    'attendance': attendance,
    'internal_marks': internal_marks,
    'assignment_score': assignment_score,
    'study_hours': study_hours,
    'previous_grade': previous_grade,
    'final_marks': all_marks,
    'dropout': dropout,
    'mental_health': mental_health
})

# Create student folder if it doesn't exist
os.makedirs('student', exist_ok=True)
data.to_csv('student/academic_performance.csv', index=False)

print('Sample dataset created successfully!')
print('Total students:', len(data))
print('Dropout distribution:\n', data['dropout'].value_counts())
print('Mental health distribution:\n', data['mental_health'].value_counts())
print('Avg CGPA:', round(data['final_marks'].mean() / 10.0, 2))
print('Avg Attendance:', round(data['attendance'].mean(), 1))
print('\nDataset saved at: student/academic_performance.csv')
