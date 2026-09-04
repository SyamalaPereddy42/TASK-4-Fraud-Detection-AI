
import os
import glob
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

# ============================================================
# FRAUD DETECTION MODEL TRAINING
# ============================================================

print("=" * 60)
print("FRAUD DETECTION AI/ML MODEL")
print("=" * 60)

# ------------------------------------------------------------
# 1. Find CSV dataset
# ------------------------------------------------------------

csv_files = glob.glob(os.path.join("data", "*.csv"))

if not csv_files:
    print("\nERROR: No CSV dataset found!")
    print("Please place your fraud detection CSV file inside:")
    print("data")
    print("\nExample:")
    print("Fraud_Detection_Project")
    print("├── data")
    print("│   └── fraud_dataset.csv")
    print("├── models")
    print("└── train_model.py")
    exit()

dataset_path = csv_files[0]

print(f"\nDataset found: {dataset_path}")

# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(dataset_path)

print(f"\nDataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 3. Find target column
# ------------------------------------------------------------

possible_targets = [
    "Class",
    "class",
    "is_fraud",
    "fraud",
    "Fraud",
    "target",
    "Target"
]

target_column = None

for column in possible_targets:
    if column in df.columns:
        target_column = column
        break

if target_column is None:
    print("\nERROR: Could not find the fraud target column.")
    print("Expected one of:")
    print(possible_targets)
    print("\nYour columns are:")
    print(df.columns.tolist())
    exit()

print(f"\nTarget column detected: {target_column}")

# ------------------------------------------------------------
# 4. Remove unnecessary columns
# ------------------------------------------------------------

# Remove completely empty columns
df = df.dropna(axis=1, how="all")

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Remove ID-like columns where possible
columns_to_remove = []

for column in X.columns:
    column_lower = column.lower()

    if column_lower in [
        "id",
        "transaction_id",
        "transactionid"
    ]:
        columns_to_remove.append(column)

if columns_to_remove:
    X = X.drop(columns=columns_to_remove)
    print(f"\nRemoved ID columns: {columns_to_remove}")

# ------------------------------------------------------------
# 5. Convert categorical columns
# ------------------------------------------------------------

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns

if len(categorical_columns) > 0:
    print("\nCategorical columns found:")
    print(list(categorical_columns))

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True
    )

# ------------------------------------------------------------
# 6. Handle missing values
# ------------------------------------------------------------

X = X.replace([float("inf"), float("-inf")], pd.NA)

for column in X.columns:
    if X[column].isnull().any():
        if pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].fillna(X[column].median())
        else:
            X[column] = X[column].fillna(0)

# ------------------------------------------------------------
# 7. Convert target to numeric
# ------------------------------------------------------------

if not pd.api.types.is_numeric_dtype(y):

    unique_values = y.dropna().unique()

    print("\nTarget values found:")
    print(unique_values)

    mapping = {
        "fraud": 1,
        "Fraud": 1,
        "yes": 1,
        "Yes": 1,
        "true": 1,
        "True": 1,
        "legitimate": 0,
        "Legitimate": 0,
        "no": 0,
        "No": 0,
        "false": 0,
        "False": 0
    }

    y = y.map(mapping)

    if y.isnull().any():
        print("\nERROR: Could not convert target values to 0/1.")
        exit()

y = y.astype(int)

# ------------------------------------------------------------
# 8. Display class distribution
# ------------------------------------------------------------

print("\nFraud class distribution:")
print(y.value_counts())

# ------------------------------------------------------------
# 9. Train-test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 10. Feature scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 11. Train Logistic Regression model
# ------------------------------------------------------------

print("\nTraining model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Model training completed!")

# ------------------------------------------------------------
# 12. Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test_scaled)
y_probability = model.predict_proba(X_test_scaled)[:, 1]

# ------------------------------------------------------------
# 13. Evaluation
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

try:
    auc = roc_auc_score(y_test, y_probability)
    print(f"\nROC-AUC Score: {auc:.4f}")
except Exception:
    print("\nROC-AUC could not be calculated.")

# ------------------------------------------------------------
# 14. Save model and scaler
# ------------------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Save feature names for Streamlit prediction
joblib.dump(
    list(X.columns),
    "models/feature_names.pkl"
)

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print("\nmodels/fraud_model.pkl")
print("models/scaler.pkl")
print("models/feature_names.pkl")

print("\nTraining completed successfully! 🚀")

