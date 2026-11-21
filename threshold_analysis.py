
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load our prepared dataset
data = pd.read_csv("data/processed/ml_dataset.csv")
data["Date"] = pd.to_datetime(data["Date"])

features = [
    "Return_1D",
    "Return_5D",
    "Volatility_10D",
    "Price_vs_MA10",
    "Volume_Change"
]

# Use the same training and validation periods
train = data[data["Date"] < "2024-12-24"]

validation = data[
    (data["Date"] >= "2025-01-01") &
    (data["Date"] < "2026-01-01")
]

X_train = train[features]
y_train = train["Risk_Label"]

X_val = validation[features]
y_val = validation["Risk_Label"]

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train the two most interesting models
models = {
    "Balanced Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
}

# Test different classification thresholds
thresholds = [
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50
]

results = []

for name, model in models.items():

    print("\nTraining:", name)

    if name == "Balanced Logistic Regression":
        model.fit(X_train_scaled, y_train)
        probabilities = model.predict_proba(
            X_val_scaled
        )[:, 1]
    else:
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(
            X_val
        )[:, 1]

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "Model": name,
            "Threshold": threshold,
            "Precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_val, predictions, zero_division=0
            ),
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp
        })

# Display results
summary = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON")
print(summary.round(4).to_string(index=False))

# Identify the best validation F1-score
best = summary.loc[summary["F1"].idxmax()]

print("\nBEST VALIDATION RESULT")
print(best.to_string())

# Save results
Path("results").mkdir(exist_ok=True)

summary.to_csv(
    "results/threshold_comparison.csv",
    index=False
)

print("\nThreshold analysis completed!")
