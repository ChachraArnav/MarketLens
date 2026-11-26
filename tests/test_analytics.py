import pandas as pd
import pytest

from marketlens import analytics


def test_total_return_and_drawdown():
    close = pd.Series([100.0, 120.0, 90.0, 110.0])
    assert analytics.total_return(close) == pytest.approx(10.0)
    assert analytics.max_drawdown(close) == pytest.approx(-25.0)
    assert analytics.drawdown(close).iloc[1] == 0


def test_buy_and_hold_drifts_without_rebalancing():
    table = pd.DataFrame({"A": [10.0, 20.0], "B": [10.0, 10.0]})
    value = analytics.buy_and_hold(table, {"A": 1, "B": 1})
    # 50 in A doubles to 100, 50 in B stays at 50.
    assert value.tolist() == [100.0, 150.0]


def test_buy_and_hold_rejects_zero_weights():
    with pytest.raises(ValueError):
        analytics.buy_and_hold(pd.DataFrame({"A": [1.0, 2.0]}), {"A": 0})
