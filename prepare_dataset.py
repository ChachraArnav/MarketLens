
import pandas as pd
import numpy as np
from pathlib import Path

# Load the original collected ETF data
data = pd.read_csv("data/processed/all_etfs.csv")

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["Ticker", "Date"]).copy()

# Calculate historical indicators separately for each ETF
groups = data.groupby("Ticker")

data["Return_1D"] = groups["Close"].pct_change(
    fill_method=None
)

data["Return_5D"] = groups["Close"].pct_change(
    periods=5, fill_method=None
)

# Historical volatility over the previous 20 days
data["Volatility_20D"] = (
    data.groupby("Ticker")["Return_1D"]
    .transform(lambda x: x.rolling(20).std())
)

# Ten-day moving average
data["MA_10"] = (
    groups["Close"].transform(
        lambda x: x.rolling(10).mean()
    )
)

data["Price_vs_MA10"] = (
    data["Close"] / data["MA_10"] - 1
)

# Change in daily trading volume
data["Volume_Change"] = groups["Volume"].pct_change(
    fill_method=None
)

# Future five-day return (used only to create labels)
data["Future_Return_5D"] = (
    groups["Close"].shift(-5) / data["Close"] - 1
)

# Fixed risk: a decline of at least 3%
data["Fixed_Risk"] = (
    data["Future_Return_5D"] <= -0.03
).astype(int)

# Risk relative to historical volatility
data["Risk_Threshold"] = (
    -1.5 * data["Volatility_20D"] * np.sqrt(5)
)

data["Adjusted_Risk"] = (
    data["Future_Return_5D"] <= data["Risk_Threshold"]
).astype(int)

# Select historical features for our models
features = [
    "Return_1D",
    "Return_5D",
    "Volatility_20D",
    "Price_vs_MA10",
    "Volume_Change"
]

# Remove rows where indicators or future outcomes
# cannot be calculated
data = data.replace([np.inf, -np.inf], np.nan)

data = data.dropna(
    subset=features + [
        "Future_Return_5D",
        "Risk_Threshold"
    ]
)

# Ensure each ETF has only one row per date
duplicates = data.duplicated(
    subset=["Ticker", "Date"]
).sum()

print("Duplicate ETF-date records:", duplicates)

# Save the final prepared dataset
Path("data/processed").mkdir(
    parents=True, exist_ok=True
)

data.to_csv(
    "data/processed/final_ml_dataset.csv",
    index=False
)

print("\nFINAL DATASET SUMMARY")
print("Total observations:", len(data))
print("Number of ETFs:", data["Ticker"].nunique())
print("Number of features:", len(features))

print("\nFixed risk distribution:")
print(data["Fixed_Risk"].value_counts(normalize=True) * 100)

print("\nAdjusted risk distribution:")
print(data["Adjusted_Risk"].value_counts(normalize=True) * 100)

print("\nFinal dataset saved successfully!")
