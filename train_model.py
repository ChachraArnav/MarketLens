
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# Load prepared observations
data = pd.read_csv("data/processed/ml_dataset.csv")
data["Date"] = pd.to_datetime(data["Date"])

features = [
    "Return_1D",
    "Return_5D",
    "Volatility_10D",
    "Price_vs_MA10",
    "Volume_Change"
]

# Split by calendar date
train_end = pd.Timestamp("2024-12-31")
validation_end = pd.Timestamp("2025-12-31")

# Five trading days separate the periods
# to reduce overlap in future-return labels.
dates = sorted(data["Date"].unique())

def date_position(target):
    return sum(date <= target for date in dates)

train_pos = date_position(train_end)
val_pos = date_position(validation_end)

train_dates = dates[:max(0, train_pos - 5)]
val_dates = dates[train_pos:max(train_pos, val_pos - 5)]
test_dates = dates[val_pos:]

train = data[data["Date"].isin(train_dates)]
validation = data[data["Date"].isin(val_dates)]
test = data[data["Date"].isin(test_dates)]

print("Training observations:", len(train))
print("Validation observations:", len(validation))
print("Testing observations:", len(test))

# Separate predictors and labels
X_train = train[features]
y_train = train["Risk_Label"]

X_val = validation[features]
y_val = validation["Risk_Label"]

X_test = test[features]
y_test = test["Risk_Label"]

# Standardize using training observations only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Evaluate the untouched test period
predictions = model.predict(X_test_scaled)

print("\nLOGISTIC REGRESSION RESULTS")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

# Save evaluation results
Path("results").mkdir(exist_ok=True)

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    "results/logistic_regression_results.csv"
)

print("\nModel evaluation completed!")
