
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix
)

# Load the prepared dataset
data = pd.read_csv("data/processed/final_ml_dataset.csv")
data["Date"] = pd.to_datetime(data["Date"])

features = [
    "Return_1D",
    "Return_5D",
    "Volatility_20D",
    "Price_vs_MA10",
    "Volume_Change"
]

# Use the same split as our consolidated evaluation
training_dates = sorted(
    data.loc[
        data["Date"] <= "2024-12-31", "Date"
    ].unique()
)

safe_dates = training_dates[:-5]

train = data[data["Date"].isin(safe_dates)]

validation = data[
    (data["Date"] >= "2025-01-01") &
    (data["Date"] <= "2025-12-31")
]

X_train = train[features]
X_val = validation[features]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

Path("results/ml_research").mkdir(
    parents=True, exist_ok=True
)

for target in ["Fixed_Risk", "Adjusted_Risk"]:

    y_train = train[target]
    y_val = validation[target]

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

        print("Evaluating:", target, "-", name)

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

        model_id = (
            target + "_" +
            name.replace(" ", "_")
        )

        # Save individual validation predictions
        predictions_df = pd.DataFrame({
            "Date": validation["Date"].values,
            "Ticker": validation["Ticker"].values,
            "Actual": y_val.values,
            "Probability": probabilities
        })

        predictions_df.to_csv(
            f"results/ml_research/{model_id}_predictions.csv",
            index=False
        )

        # Calculate the precision-recall curve
        precision, recall, thresholds = (
            precision_recall_curve(
                y_val, probabilities
            )
        )

        curve_df = pd.DataFrame({
            "Precision": precision,
            "Recall": recall
        })

        curve_df.to_csv(
            f"results/ml_research/{model_id}_curve.csv",
            index=False
        )

        print(
            "Average Precision:",
            round(
                average_precision_score(
                    y_val, probabilities
                ), 4
            )
        )

print("\nML research data saved successfully!")
