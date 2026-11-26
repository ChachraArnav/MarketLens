"""Downloading, combining and validating the raw price data."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from . import config


def download(
    tickers: list[str] | None = None,
    start: str = config.START_DATE,
    end: str | None = None,
    raw_dir: Path = config.RAW_DIR,
) -> pd.DataFrame:
    """Download daily adjusted prices from Yahoo Finance into one CSV per ticker."""
    import yfinance as yf  # imported here so the rest of the package works offline

    raw_dir.mkdir(parents=True, exist_ok=True)
    end = end or date.today().isoformat()
    summary = []
    for ticker in tickers or config.TICKERS:
        frame = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            multi_level_index=False,
            progress=False,
        )
        if frame.empty:
            print(f"  {ticker}: no data returned")
            continue
        frame = frame.reset_index()
        frame["Ticker"] = ticker
        frame.to_csv(raw_dir / f"{ticker}.csv", index=False)
        summary.append(
            {
                "Ticker": ticker,
                "Rows": len(frame),
                "Missing_Values": int(frame.isna().sum().sum()),
                "First_Date": str(frame["Date"].min().date()),
                "Last_Date": str(frame["Date"].max().date()),
            }
        )
        print(f"  {ticker}: {len(frame)} rows")
    result = pd.DataFrame(summary)
    result.to_csv(raw_dir.parent / "collection_summary.csv", index=False)
    return result


def load_raw(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Combine the per-ticker CSVs into one long table sorted by ticker and date."""
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {raw_dir}. Run `python -m marketlens download`.")
    prices = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    prices["Date"] = pd.to_datetime(prices["Date"])
    return prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def validate(prices: pd.DataFrame) -> list[str]:
    """Return a list of data problems. An empty list means the data passed."""
    problems = []
    required = {"Date", "Ticker", "Close", "Volume"}
    missing = required - set(prices.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    if (prices["Close"] <= 0).any():
        problems.append(f"{int((prices['Close'] <= 0).sum())} non-positive closing prices")
    duplicates = int(prices.duplicated(subset=["Ticker", "Date"]).sum())
    if duplicates:
        problems.append(f"{duplicates} duplicate ticker-date rows")
    nulls = int(prices[list(required)].isna().sum().sum())
    if nulls:
        problems.append(f"{nulls} missing values in required columns")
    return problems
