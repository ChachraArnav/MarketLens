
import pandas as pd

data = pd.read_csv("data/processed/all_etfs.csv")

# Check for invalid prices
print("Invalid closing prices:")
print((data["Close"] <= 0).sum())

# Check the starting and ending prices
summary = []

for ticker, group in data.groupby("Ticker"):
    group = group.sort_values("Date")

    first_price = group["Close"].iloc[0]
    last_price = group["Close"].iloc[-1]

    growth = (last_price / first_price - 1) * 100

    summary.append({
        "ETF": ticker,
        "First_Price": round(first_price, 2),
        "Last_Price": round(last_price, 2),
        "Total_Return_%": round(growth, 2)
    })

results = pd.DataFrame(summary)

print("\nPRICE VALIDATION SUMMARY")
print(results.to_string(index=False))
