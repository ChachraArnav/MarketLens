"""Chronological train / validation / test periods with a purge gap."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(data: pd.DataFrame, horizon: int = config.HORIZON) -> Splits:
    """Split by calendar date.

    The last `horizon` trading dates of the training period are dropped,
    because their labels look into the validation period. Validation is 2025;
    test is everything from 2026 on and is only scored once, at the end.
    """
    # Date-then-ticker order, as in the original research. Random Forest
    # bootstrapping and the neural network's shuffling depend on row order.
    data = data.assign(Date=pd.to_datetime(data["Date"])).sort_values(["Date", "Ticker"])
    dates = data["Date"]
    train_dates = sorted(dates[dates <= config.TRAIN_END].unique())
    safe_train_dates = set(train_dates[:-horizon]) if horizon else set(train_dates)

    train = data[dates.isin(safe_train_dates)]
    validation = data[(dates >= config.VALIDATION_START) & (dates <= config.VALIDATION_END)]
    test = data[dates >= config.TEST_START]
    return Splits(train=train, validation=validation, test=test)
