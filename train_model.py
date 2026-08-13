import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# ==========================================
# 1. LOAD DATASETS
# ==========================================

files = [
    "dataset/A.csv",
    "dataset/B.csv",
    "dataset/C.csv",
    "dataset/D.csv",
    "dataset/E.csv",
    "dataset/F.csv",
    "dataset/G.csv",
    "dataset/H.csv",
    "dataset/I.csv",
    "dataset/J.csv",
    "dataset/K.csv",
    "dataset/L.csv",
    "dataset/M.csv",
    "dataset/N.csv",
    "dataset/O.csv",
    "dataset/P.csv",
    "dataset/Q.csv",
    "dataset/R.csv",
    "dataset/S.csv",
    "dataset/T.csv",
    "dataset/U.csv",
    "dataset/V.csv",
    "dataset/W.csv",
    "dataset/X.csv",
    "dataset/Y.csv",
    "dataset/Z.csv",
]

dataframes = []

for file in files:

    df = pd.read_csv(file)

    dataframes.append(df)

    print(f"Loaded {file}: {len(df)} samples")


# ==========================================
# 2. COMBINE DATASETS
# ==========================================

data = pd.concat(
    dataframes,
    ignore_index=True
)

print()
print("Total samples:", len(data))


# ==========================================
# 3. SEPARATE FEATURES AND LABEL
# ==========================================

X = data.drop(
    columns=["label"]
)

y = data["label"]


print("Number of features:", X.shape[1])
print("Number of classes:", y.nunique())

print()
print("Classes:")
print(sorted(y.unique()))


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. CREATE RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 6. TRAIN
# ==========================================

print()
print("Training model...")

model.fit(
    X_train,
    y_train
)

print("Training complete!")


# ==========================================
# 7. CHECK LEARNED CLASSES
# ==========================================

print()
print("Model learned these classes:")
print(model.classes_)


# ==========================================
# 8. PREDICT TEST DATA
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 9. ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print("================================")
print("MODEL RESULTS")
print("================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

print()
print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ==========================================
# 12. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "sign_model.pkl"
)

print()
print("Model saved as sign_model.pkl")