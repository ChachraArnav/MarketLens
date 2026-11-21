
import pandas as pd
from pathlib import Path

from sklearn.dummy import DummyClassifier
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

# Load our machine learning dataset
data = pd.read_csv("data/processed/ml_dataset.csv")
data["Date"] = pd.to_datetime(data["Date"])

features = [
    "Return_1D",
    "Return_5D",
    "Volatility_10D",
    "Price_vs_MA10",
    "Volume_Change"
]

# Sort observations by date
data = data.sort_values(["Date", "Ticker"])

# Use earlier years for training
# and 2025 for model development
train = data[data["Date"] < "2024-12-24"]
validation = data[
    (data["Date"] >= "2025-01-01") &
    (data["Date"] < "2026-01-01")
]

X_train = train[features]
y_train = train["Risk_Label"]

X_val = validation[features]
y_val = validation["Risk_Label"]

print("Training observations:", len(train))
print("Validation observations:", len(validation))

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define our models
models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

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

results = []

for name, model in models.items():

    print("\nTraining:", name)

    # Logistic Regression uses scaled features
    if "Logistic Regression" in name:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_val_scaled)
        probabilities = model.predict_proba(X_val_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
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
        )
    })

# Compare model results
summary = pd.DataFrame(results)
summary = summary.sort_values(
    "Average_Precision",
    ascending=False
)

print("\nMODEL COMPARISON")
print(summary.round(4).to_string(index=False))

# Save the results
Path("results").mkdir(exist_ok=True)

summary.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nModel comparison completed!")
