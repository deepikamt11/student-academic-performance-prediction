
import pandas as pd

df = pd.read_csv("student/student-mat.csv", sep=";")

print("Dataset Shape:", df.shape)
print("\nAvailable Columns:")
print(df.columns.tolist())

print("\nFirst 5 Student Records:")
print(df.head())
print("\nStudy Time Values:")
print(sorted(df["studytime"].unique()))

print("\nAbsence Range:")
print(df["absences"].min(), "to", df["absences"].max())

print("\nGrade Range:")
print(df[["G1", "G2", "G3"]].min())
print(df[["G1", "G2", "G3"]].max())