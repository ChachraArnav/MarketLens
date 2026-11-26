import pandas as pd

from marketlens.data import validate


def test_clean_data_passes(prices):
    assert validate(prices) == []


def test_validation_catches_bad_rows(prices):
    bad = prices.copy()
    bad.loc[0, "Close"] = -1
    bad = pd.concat([bad, bad.iloc[[5]]])
    problems = validate(bad)
    assert any("non-positive" in p for p in problems)
    assert any("duplicate" in p for p in problems)
