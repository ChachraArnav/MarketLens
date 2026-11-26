import pandas as pd

from marketlens import config
from marketlens.splits import chronological_split


def make(dates):
    return pd.DataFrame({"Date": dates, "Ticker": "AAA.TO", "x": range(len(dates))})


def test_periods_do_not_overlap_and_are_in_order():
    data = make(pd.bdate_range("2024-06-01", "2026-03-31"))
    s = chronological_split(data)
    assert s.train["Date"].max() < s.validation["Date"].min()
    assert s.validation["Date"].max() < s.test["Date"].min()
    assert s.validation["Date"].min() >= pd.Timestamp(config.VALIDATION_START)
    assert s.test["Date"].min() >= pd.Timestamp(config.TEST_START)


def test_training_is_purged_before_validation():
    dates = pd.bdate_range("2024-06-01", "2025-06-30")
    s = chronological_split(make(dates))
    train_dates = dates[dates <= config.TRAIN_END]
    # The last HORIZON training dates are removed because their labels peek into 2025.
    assert s.train["Date"].max() == train_dates[-config.HORIZON - 1]
    assert len(s.train) == len(train_dates) - config.HORIZON
