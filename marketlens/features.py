"""Model inputs and risk labels.

Features at row t use only prices up to and including day t. Labels use the
next HORIZON days and exist only to train and score models; they are never
fed back in as inputs. tests/test_features.py checks both properties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def add_features(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.sort_values(["Ticker", "Date"]).copy()
    by_ticker = data.groupby("Ticker")
    data["Return_1D"] = by_ticker["Close"].pct_change(fill_method=None)
    data["Return_5D"] = by_ticker["Close"].pct_change(periods=5, fill_method=None)
    data["Volatility_20D"] = data.groupby("Ticker")["Return_1D"].transform(
        lambda r: r.rolling(20).std()
    )
    data["MA_10"] = by_ticker["Close"].transform(lambda c: c.rolling(10).mean())
    data["Price_vs_MA10"] = data["Close"] / data["MA_10"] - 1
    data["Volume_Change"] = by_ticker["Volume"].pct_change(fill_method=None)
    return data


def add_labels(data: pd.DataFrame, horizon: int = config.HORIZON) -> pd.DataFrame:
    data = data.sort_values(["Ticker", "Date"]).copy()
    future_close = data.groupby("Ticker")["Close"].shift(-horizon)
    data["Future_Return_5D"] = future_close / data["Close"] - 1
    data["Fixed_Risk"] = (data["Future_Return_5D"] <= config.FIXED_DROP).astype(int)
    data["Risk_Threshold"] = -config.ADJUSTED_MULTIPLIER * data["Volatility_20D"] * np.sqrt(horizon)
    data["Adjusted_Risk"] = (data["Future_Return_5D"] <= data["Risk_Threshold"]).astype(int)
    return data


def build_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Features + labels, keeping only rows where every input and outcome is known."""
    data = add_labels(add_features(prices))
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=config.FEATURES + ["Future_Return_5D", "Risk_Threshold"])
    return data.reset_index(drop=True)
