
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import date

# ETFs selected for our research
etfs = [
    "VFV.TO", "XEQT.TO", "ZNQ.TO",
    "XIC.TO", "XEF.TO", "XEC.TO",
    "XFN.TO", "XEG.TO", "XBB.TO",
    "XGD.TO", "XRE.TO", "XIT.TO",
    "CGL.TO"
]

# Create the data folder if needed
folder = Path("data/raw")
folder.mkdir(parents=True, exist_ok=True)

start_date = "2021-01-01"
end_date = date.today().isoformat()

summary = []

for ticker in etfs:
    print("Downloading:", ticker)

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            multi_level_index=False,
            progress=False
        )

        if data.empty:
            print("No data found for", ticker)
            continue

        data = data.reset_index()
        data["Ticker"] = ticker

        # Save each ETF separately
        filename = folder / f"{ticker}.csv"
        data.to_csv(filename, index=False)

        summary.append({
            "Ticker": ticker,
            "Rows": len(data),
            "Missing_Values": int(data.isnull().sum().sum()),
            "First_Date": str(data["Date"].min().date()),
            "Last_Date": str(data["Date"].max().date())
        })

        print("Saved", len(data), "rows")

    except Exception as error:
        print("Error with", ticker, ":", error)

# Save a summary of our collection
summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    "data/collection_summary.csv",
    index=False
)

print("\nDATA COLLECTION SUMMARY")
print(summary_df)
