
import pandas as pd
from pathlib import Path

# Load historical ETF observations
data = pd.read_csv("data/processed/all_etfs.csv")

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["Ticker", "Date"]).copy()

# Calculate features separately for each ETF
groups = data.groupby("Ticker")

data["Return_1D"] = groups["Close"].pct_change()
data["Return_5D"] = groups["Close"].pct_change(5)

data["Volatility_10D"] = (
    data.groupby("Ticker")["Return_1D"]
    .transform(lambda x: x.rolling(10).std())
)

data["MA_10"] = (
    groups["Close"].transform(
        lambda x: x.rolling(10).mean()
    )
)

# Compare price with its moving average
data["Price_vs_MA10"] = data["Close"] / data["MA_10"] - 1

# Calculate daily volume change
data["Volume_Change"] = groups["Volume"].pct_change()

# Calculate the future five-day return
data["Future_Return_5D"] = (
    groups["Close"].shift(-5) / data["Close"] - 1
)

# Create our classification label
data["Risk_Label"] = (
    data["Future_Return_5D"] <= -0.03
).astype(int)

# Remove rows with missing feature values
# and rows without a known future outcome
features = [
    "Return_1D",
    "Return_5D",
    "Volatility_10D",
    "Price_vs_MA10",
    "Volume_Change"
]

data = data.dropna(
    subset=features + ["Future_Return_5D"]
)

# Remove infinite values caused by division by zero
import numpy as np

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=features)

# Save the machine learning dataset
Path("data/processed").mkdir(parents=True, exist_ok=True)

data.to_csv(
    "data/processed/ml_dataset.csv",
    index=False
)

print("\nMACHINE LEARNING DATASET")
print("Total observations:", len(data))

print("\nRisk label distribution:")
print(data["Risk_Label"].value_counts())

print("\nRisk label percentages:")
print(data["Risk_Label"].value_counts(normalize=True) * 100)

print("\nDataset saved successfully!")
