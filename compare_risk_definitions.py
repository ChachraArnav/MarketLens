
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# Load the dataset containing both risk definitions
data = pd.read_csv(
    "data/processed/volatility_risk_dataset.csv"
)

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["Date", "Ticker"])

# Create the same historical features for both models
data["Return_1D"] = (
    data.groupby("Ticker")["Close"].pct_change()
)

data["Return_5D"] = (
    data.groupby("Ticker")["Close"].pct_change(5)
)

data["Price_vs_MA10"] = (
    data["Close"] /
    data.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(10).mean()
    ) - 1
)

data["Volume_Change"] = (
    data.groupby("Ticker")["Volume"].pct_change()
)

# Use the existing 20-day volatility feature
features = [
    "Return_1D",
    "Return_5D",
    "Volatility_20D",
    "Price_vs_MA10",
    "Volume_Change"
]

# Remove invalid feature values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=features)

# Use 2021-2024 for training and 2025 for validation
# Leave a small gap for overlapping five-day labels
train = data[data["Date"] < "2024-12-24"]

validation = data[
    (data["Date"] >= "2025-01-01") &
    (data["Date"] < "2026-01-01")
]

print("Training observations:", len(train))
print("Validation observations:", len(validation))

X_train = train[features]
X_val = validation[features]

# Scale inputs using training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

results = []

# Repeat the experiment for each risk definition
for target in ["Fixed_Risk", "Adjusted_Risk"]:

    print("\nRisk definition:", target)

    y_train = train[target]
    y_val = validation[target]

    print(
        "Validation risk frequency:",
        round(y_val.mean() * 100, 2), "%"
    )

    models = {
        "Logistic Regression": LogisticRegression(
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

    for name, model in models.items():

        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            probabilities = model.predict_proba(
                X_val_scaled
            )[:, 1]
        else:
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(
                X_val
            )[:, 1]

        # Use a common threshold for initial comparison
        predictions = (probabilities >= 0.5).astype(int)

        results.append({
            "Risk_Definition": target,
            "Model": name,
            "Accuracy": accuracy_score(
                y_val, predictions
            ),
            "Precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_val, predictions, zero_division=0
            ),
            "Average_Precision": average_precision_score(
                y_val, probabilities
            ),
            "Baseline_AP": y_val.mean()
        })

# Display comparison
summary = pd.DataFrame(results)

print("\nRISK DEFINITION MODEL COMPARISON")
print(summary.round(4).to_string(index=False))

# Save the results
Path("results").mkdir(exist_ok=True)

summary.to_csv(
    "results/risk_definition_model_comparison.csv",
    index=False
)

print("\nRisk definition comparison completed!")
