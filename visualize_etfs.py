
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create output folder
Path("results").mkdir(exist_ok=True)

# Load the datasets
data = pd.read_csv("data/processed/etfs_with_returns.csv")
summary = pd.read_csv("results/etf_statistics.csv")

data["Date"] = pd.to_datetime(data["Date"])

# ---------------------------------------
# Chart 1: Annualized volatility
# ---------------------------------------

plt.figure(figsize=(12, 6))

sns.barplot(
    data=summary,
    x="ETF",
    y="Annual_Volatility_%"
)

plt.title("Annualized Volatility of ETFs")
plt.xlabel("ETF")
plt.ylabel("Annualized Volatility (%)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("results/volatility_comparison.png")
plt.close()

# ---------------------------------------
# Chart 2: Growth of $100
# ---------------------------------------

plt.figure(figsize=(12, 6))

for ticker, group in data.groupby("Ticker"):
    group = group.sort_values("Date")

    growth = 100 * group["Close"] / group["Close"].iloc[0]

    plt.plot(
        group["Date"],
        growth,
        label=ticker
    )

plt.title("Historical Growth of $100 by ETF")
plt.xlabel("Date")
plt.ylabel("Investment Value ($)")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()

plt.savefig("results/growth_comparison.png")
plt.close()

# ---------------------------------------
# Chart 3: ETF correlation heatmap
# ---------------------------------------

returns = data.pivot(
    index="Date",
    columns="Ticker",
    values="Daily_Return"
)

correlation = returns.corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between ETF Daily Returns")
plt.tight_layout()

plt.savefig("results/correlation_heatmap.png")
plt.close()

print("All three charts saved in the results folder!")
