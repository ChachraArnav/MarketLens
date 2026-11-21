from pathlib import Path
import pandas as pd
import numpy as np

# Load our combined ETF dataset
data = pd.read_csv("data/processed/all_etfs.csv")

# Convert dates and sort the observations
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["Ticker", "Date"])

# Calculate daily percentage returns
data["Daily_Return"] = data.groupby("Ticker")["Close"].pct_change()

# Calculate statistics for each ETF
results = []

for ticker, group in data.groupby("Ticker"):
    returns = group["Daily_Return"].dropna()

    total_return = (
        group["Close"].iloc[-1] / group["Close"].iloc[0] - 1
    ) * 100

    volatility = returns.std() * np.sqrt(252) * 100

    results.append({
        "ETF": ticker,
        "Total_Return_%": round(total_return, 2),
        "Annual_Volatility_%": round(volatility, 2),
        "Average_Daily_Return_%": round(returns.mean() * 100, 4)
    })

# Display the results
summary = pd.DataFrame(results)
summary = summary.sort_values(
    "Annual_Volatility_%",
    ascending=False
)

print("\nETF PERFORMANCE SUMMARY")
print(summary.to_string(index=False))


# Create the results folder if it does not exist
Path("results").mkdir(parents=True, exist_ok=True)

# Save the analysis
summary.to_csv("results/etf_statistics.csv", index=False)
data.to_csv("data/processed/etfs_with_returns.csv", index=False)

print("\nAnalysis completed!")
