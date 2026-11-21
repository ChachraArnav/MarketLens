
import pandas as pd
from pathlib import Path

# Find all downloaded ETF files
files = sorted(Path("data/raw").glob("*.csv"))

print("Number of ETF files:", len(files))

# Combine the individual ETF datasets
all_data = []

for file in files:
    df = pd.read_csv(file)
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Convert dates to datetime
data["Date"] = pd.to_datetime(data["Date"])

print("\nDataset shape:", data.shape)
print("\nColumns:", data.columns.tolist())
print("\nFirst five rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())

print("\nNumber of records per ETF:")
print(data.groupby("Ticker").size())

print("\nDuplicate ticker-date records:")
print(data.duplicated(subset=["Ticker", "Date"]).sum())

# Save combined dataset
Path("data/processed").mkdir(parents=True, exist_ok=True)
data.to_csv("data/processed/all_etfs.csv", index=False)

print("\nCombined dataset saved!")
