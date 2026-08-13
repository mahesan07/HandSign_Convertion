import pandas as pd


file = "dataset/A.csv"

# Load A dataset
df = pd.read_csv(file)

print("Before:")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# Create the correct feature names
feature_names = [
    f"feature_{i}"
    for i in range(63)
]

# Add label
new_columns = feature_names + ["label"]

# Rename columns
df.columns = new_columns

# Save corrected dataset
df.to_csv(file, index=False)

print()
print("A.csv columns fixed!")

print("Rows:", len(df))
print("Columns:", len(df.columns))