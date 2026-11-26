import pandas as pd
import pytest

from marketlens.crosscheck import compare


def make_sources(distribution=0.10, ex_index=6, n=12):
    """TMX = traded prices; Yahoo = the same prices back-adjusted for one distribution."""
    dates = pd.bdate_range("2025-03-17", periods=n)
    tmx_close = pd.Series([30 + 0.1 * i for i in range(n)], index=dates)
    factor = pd.Series(1.0, index=dates)
    factor.iloc[:ex_index] = 1 - distribution / tmx_close.iloc[ex_index - 1]
    volume = pd.Series(range(1000, 1000 + n), index=dates)
    tmx = pd.DataFrame({"Date": dates, "Close": tmx_close.values, "Volume": volume.values})
    yahoo = pd.DataFrame(
        {"Date": dates, "Close": (tmx_close * factor).values, "Volume": volume.values}
    )
    return yahoo, tmx, dates


def test_identical_sources_agree():
    yahoo, tmx, _ = make_sources(distribution=0.0)
    daily, s = compare(yahoo, tmx)
    assert s["dates_only_in_tmx"] == [] and s["dates_only_in_yahoo"] == []
    assert s["max_close_difference_cents"] == 0
    assert s["volume_within_rounding_share"] == 1
    assert s["distribution_steps"] == []


def test_distribution_is_found_on_its_ex_date_with_its_amount():
    yahoo, tmx, dates = make_sources(distribution=0.09, ex_index=6)
    daily, s = compare(yahoo, tmx)
    assert len(s["distribution_steps"]) == 1
    step = s["distribution_steps"][0]
    assert step["Ex_Date"] == str(dates[6].date())
    assert step["Implied_Cash_Per_Unit"] == pytest.approx(0.09, abs=1e-4)
    # Once the adjustment is undone, prices match exactly.
    assert s["max_close_difference_cents"] == pytest.approx(0, abs=1e-6)


def test_missing_and_mispriced_days_are_reported():
    yahoo, tmx, dates = make_sources(distribution=0.0)
    yahoo = yahoo.drop(index=3)
    yahoo.loc[8, "Close"] += 0.02
    daily, s = compare(yahoo, tmx)
    assert s["dates_only_in_tmx"] == [str(dates[3].date())]
    assert s["max_close_difference_cents"] == pytest.approx(2, abs=1e-6)
    assert s["share_within_one_cent"] < 1
