import numpy as np
import pandas as pd

from marketlens import config
from marketlens.features import add_features, add_labels, build_dataset


def test_features_never_use_future_prices(prices):
    """Changing every price after day t must not change any feature at day t."""
    cutoff = prices["Date"].sort_values().unique()[200]
    before = add_features(prices)

    shocked = prices.copy()
    later = shocked["Date"] > cutoff
    shocked.loc[later, "Close"] *= 3.0
    shocked.loc[later, "Volume"] *= 7
    after = add_features(shocked)

    known = before["Date"] <= cutoff
    pd.testing.assert_frame_equal(
        before.loc[known, config.FEATURES].reset_index(drop=True),
        after.loc[known, config.FEATURES].reset_index(drop=True),
    )


def test_features_are_computed_per_ticker(prices):
    data = add_features(prices)
    first_rows = data.groupby("Ticker").head(1)
    # The first day of each ETF has no previous price, so no return.
    assert first_rows["Return_1D"].isna().all()


def test_label_matches_five_day_future_return(prices):
    data = add_labels(add_features(prices))
    one = data[data["Ticker"] == "AAA.TO"].reset_index(drop=True)
    i = 100
    expected = one.loc[i + config.HORIZON, "Close"] / one.loc[i, "Close"] - 1
    assert np.isclose(one.loc[i, "Future_Return_5D"], expected)
    assert one.loc[i, "Fixed_Risk"] == int(expected <= config.FIXED_DROP)


def test_last_days_have_no_label_and_are_dropped(prices):
    labelled = add_labels(add_features(prices))
    tail = labelled.groupby("Ticker").tail(config.HORIZON)
    assert tail["Future_Return_5D"].isna().all()

    dataset = build_dataset(prices)
    assert dataset[config.FEATURES + ["Future_Return_5D"]].notna().all().all()
    assert not np.isinf(dataset[config.FEATURES]).any().any()


def test_adjusted_threshold_scales_with_volatility(prices):
    data = build_dataset(prices)
    expected = -config.ADJUSTED_MULTIPLIER * data["Volatility_20D"] * np.sqrt(config.HORIZON)
    assert np.allclose(data["Risk_Threshold"], expected)
