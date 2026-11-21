
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

# Load our final prepared dataset
data = pd.read_csv("data/processed/final_ml_dataset.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["Date", "Ticker"])

features = [
    "Return_1D",
    "Return_5D",
    "Volatility_20D",
    "Price_vs_MA10",
    "Volume_Change"
]

# Chronological training and validation periods
train_end = pd.Timestamp("2024-12-31")
val_start = pd.Timestamp("2025-01-01")
val_end = pd.Timestamp("2025-12-31")

# Remove the final five trading dates from training
# because their future labels could cross the boundary
training_dates = sorted(
    data.loc[data["Date"] <= train_end, "Date"].unique()
)

safe_training_dates = training_dates[:-5]

train = data[data["Date"].isin(safe_training_dates)]

validation = data[
    (data["Date"] >= val_start) &
    (data["Date"] <= val_end)
]

print("Training observations:", len(train))
print("Validation observations:", len(validation))

X_train = train[features]
X_val = validation[features]

# Fit scaling only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

results = []

for target in ["Fixed_Risk", "Adjusted_Risk"]:

    y_train = train[target]
    y_val = validation[target]

    models = {
        "Baseline": DummyClassifier(
            strategy="most_frequent"
        ),

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

        print("Training:", target, "-", name)

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

# Display and save the results
summary = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(summary.round(4).to_string(index=False))

Path("results").mkdir(exist_ok=True)

summary.to_csv(
    "results/final_model_comparison.csv",
    index=False
)

print("\nEvaluation completed successfully!")
