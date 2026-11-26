"""Descriptive statistics used by the dashboard and the pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(close: pd.Series) -> float:
    """Percentage change from the first to the last price."""
    return float((close.iloc[-1] / close.iloc[0] - 1) * 100)


def annualized_volatility(close: pd.Series) -> float:
    returns = close.pct_change(fill_method=None).dropna()
    return float(returns.std() * np.sqrt(TRADING_DAYS) * 100)


def drawdown(close: pd.Series) -> pd.Series:
    """Percentage below the running peak at each date (0 at a new high)."""
    return (close / close.cummax() - 1) * 100


def max_drawdown(close: pd.Series) -> float:
    return float(drawdown(close).min())


def price_table(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Wide table of closing prices on dates every selected ticker traded."""
    return (
        prices[prices["Ticker"].isin(tickers)]
        .pivot(index="Date", columns="Ticker", values="Close")
        .sort_index()
        .dropna()
    )


def buy_and_hold(
    table: pd.DataFrame, weights: dict[str, float], start_value: float = 100.0
) -> pd.Series:
    """Value of a portfolio bought on the first date and never rebalanced."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    normalized = pd.Series({k: v / total for k, v in weights.items()})
    growth = table[list(normalized.index)].div(table.iloc[0])
    return growth.mul(normalized, axis=1).sum(axis=1) * start_value


def summary_table(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, group in prices.sort_values("Date").groupby("Ticker"):
        close = group.set_index("Date")["Close"]
        rows.append(
            {
                "ETF": ticker,
                "Total_Return_%": round(total_return(close), 2),
                "Annual_Volatility_%": round(annualized_volatility(close), 2),
                "Maximum_Drawdown_%": round(max_drawdown(close), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("Annual_Volatility_%", ascending=False)
