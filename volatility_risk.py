
import pandas as pd
import numpy as np
from pathlib import Path

# Load the original ETF observations
data = pd.read_csv("data/processed/all_etfs.csv")

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["Ticker", "Date"]).copy()

# Calculate daily returns for each ETF
data["Daily_Return"] = (
    data.groupby("Ticker")["Close"].pct_change()
)

# Calculate volatility using the past 20 days
data["Volatility_20D"] = (
    data.groupby("Ticker")["Daily_Return"]
    .transform(lambda x: x.rolling(20).std())
)

# Calculate the future five-day return
data["Future_Return_5D"] = (
    data.groupby("Ticker")["Close"].shift(-5)
    / data["Close"] - 1
)

# Fixed risk threshold: decline of at least 3%
data["Fixed_Risk"] = (
    data["Future_Return_5D"] <= -0.03
).astype(int)

# Volatility-adjusted threshold
data["Risk_Threshold"] = (
    -1.5 * data["Volatility_20D"] * np.sqrt(5)
)

data["Adjusted_Risk"] = (
    data["Future_Return_5D"] <= data["Risk_Threshold"]
).astype(int)

# Keep only observations with valid information
data = data.dropna(
    subset=[
        "Volatility_20D",
        "Future_Return_5D",
        "Risk_Threshold"
    ]
)

# Display overall risk-label distributions
print("\nFIXED 3% RISK DISTRIBUTION")
print(data["Fixed_Risk"].value_counts(normalize=True) * 100)

print("\nVOLATILITY-ADJUSTED RISK DISTRIBUTION")
print(data["Adjusted_Risk"].value_counts(normalize=True) * 100)

# Compare risk-event frequency for each ETF
summary = data.groupby("Ticker")[
    ["Fixed_Risk", "Adjusted_Risk"]
].mean() * 100

summary = summary.round(2)

print("\nRISK FREQUENCY BY ETF")
print(summary.to_string())

# Save results
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)

data.to_csv(
    "data/processed/volatility_risk_dataset.csv",
    index=False
)

summary.to_csv(
    "results/risk_definition_comparison.csv"
)

print("\nVolatility risk analysis completed!")
