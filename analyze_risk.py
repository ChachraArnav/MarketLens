
import pandas as pd
from pathlib import Path

# Load the historical ETF observations
data = pd.read_csv("data/processed/etfs_with_returns.csv")

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["Ticker", "Date"])

results = []

for ticker, group in data.groupby("Ticker"):

    # Highest price reached up to each date
    running_peak = group["Close"].cummax()

    # Percentage decline from the previous peak
    drawdown = (group["Close"] / running_peak - 1) * 100

    # Largest historical decline
    max_drawdown = drawdown.min()

    results.append({
        "ETF": ticker,
        "Maximum_Drawdown_%": round(max_drawdown, 2)
    })

summary = pd.DataFrame(results)
summary = summary.sort_values("Maximum_Drawdown_%")

print("\nMAXIMUM DRAWDOWN COMPARISON")
print(summary.to_string(index=False))

Path("results").mkdir(exist_ok=True)
summary.to_csv("results/drawdown_summary.csv", index=False)

print("\nDrawdown analysis completed!")
