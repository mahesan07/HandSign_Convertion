import pandas as pd


# Load dataset
df = pd.read_csv("dataset/A.csv")


print("================================")
print("DATASET INFORMATION")
print("================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print()


print("================================")
print("LABEL COUNTS")
print("================================")

print(df["label"].value_counts())

print()


print("================================")
print("MISSING VALUES")
print("================================")

print(df.isnull().sum().sum())

print()


print("================================")
print("DUPLICATE ROWS")
print("================================")

print(df.duplicated().sum())

print()


print("================================")
print("FEATURE TYPES")
print("================================")

print(df.dtypes.value_counts())