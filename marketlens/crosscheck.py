"""Cross-check the Yahoo Finance prices against an independent source (TMX Money).

Yahoo prices in this project are dividend-adjusted; TMX prices are the actual
traded prices. So for any one ETF:

    yahoo_adjusted_close = adjustment_factor x tmx_close

where the factor is constant between distributions and steps up on each
ex-distribution date. The check therefore asks:

  1. Do both sources have exactly the same trading days?
  2. Between distributions, is the ratio constant (prices agree)?
  3. Do the ratio steps fall on the issuer's announced ex-dates, and does the
     size of each step imply the announced cash distribution?
  4. Do daily volumes agree (allowing for Yahoo rounding them to 100 shares)?
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config

EXTERNAL_DIR = config.ROOT / "data" / "external"
CROSSCHECK_DIR = config.RESULTS_DIR / "crosscheck"

# A distribution of even 0.1% of the price moves the ratio by 1e-3, while a
# half-cent rounding difference on a ~$30 ETF moves it by under 2e-4.
STEP_TOLERANCE = 1e-3


def load_tmx(path: Path) -> pd.DataFrame:
    """Read a TMX Money 'Price History' CSV export."""
    raw = pd.read_csv(path)
    return (
        pd.DataFrame(
            {
                "Date": pd.to_datetime(raw["Date"], format="%b %d, %Y %I:%M %p"),
                "Close": raw["Close ($)"].astype(float),
                "Volume": raw["Volume"].astype("int64"),
            }
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )


def compare(yahoo: pd.DataFrame, tmx: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare one window. Both frames need Date, Close and Volume columns."""
    start, end = tmx["Date"].min(), tmx["Date"].max()
    yahoo = yahoo[(yahoo["Date"] >= start) & (yahoo["Date"] <= end)]
    y = yahoo.set_index("Date")[["Close", "Volume"]]
    t = tmx.set_index("Date")[["Close", "Volume"]]

    only_tmx = sorted(t.index.difference(y.index))
    only_yahoo = sorted(y.index.difference(t.index))
    joined = y.join(t, how="inner", suffixes=("_Yahoo", "_TMX"))

    ratio = joined["Close_Yahoo"] / joined["Close_TMX"]
    is_step = ratio.pct_change().abs() > STEP_TOLERANCE
    segment = is_step.cumsum()
    factor = ratio.groupby(segment).transform("median")

    daily = pd.DataFrame(
        {
            "Date": joined.index,
            "Yahoo_Adjusted_Close": joined["Close_Yahoo"].values,
            "TMX_Close": joined["Close_TMX"].values,
            "Adjustment_Factor": factor.values,
            "Yahoo_Unadjusted_Close": (joined["Close_Yahoo"] / factor).values,
            "Yahoo_Return_%": joined["Close_Yahoo"].pct_change().mul(100).values,
            "TMX_Return_%": joined["Close_TMX"].pct_change().mul(100).values,
            "Distribution_Step": is_step.values,
            "Yahoo_Volume": joined["Volume_Yahoo"].values,
            "TMX_Volume": joined["Volume_TMX"].values,
        }
    )
    daily["Close_Difference_Cents"] = (
        (daily["Yahoo_Unadjusted_Close"] - daily["TMX_Close"]) * 100
    ).round(4)

    steps = []
    for i in daily.index[daily["Distribution_Step"]]:
        before, after = daily.at[i - 1, "Adjustment_Factor"], daily.at[i, "Adjustment_Factor"]
        steps.append(
            {
                "Ex_Date": str(daily.at[i, "Date"].date()),
                "Implied_Cash_Per_Unit": round(
                    float(daily.at[i - 1, "TMX_Close"] * (1 - before / after)), 4
                ),
            }
        )

    ordinary = daily[~daily["Distribution_Step"]].dropna(subset=["TMX_Return_%"])
    return_gap = (ordinary["Yahoo_Return_%"] - ordinary["TMX_Return_%"]).abs()
    cents = daily["Close_Difference_Cents"].abs()
    volume_gap = (daily["Yahoo_Volume"] - daily["TMX_Volume"]).abs()
    volume_gap_pct = volume_gap / daily["TMX_Volume"] * 100
    summary = {
        "start": str(start.date()),
        "end": str(end.date()),
        "tmx_days": int(len(t)),
        "yahoo_days": int(len(y)),
        "dates_only_in_tmx": [str(d.date()) for d in only_tmx],
        "dates_only_in_yahoo": [str(d.date()) for d in only_yahoo],
        "max_close_difference_cents": round(float(cents.max()), 2),
        "share_within_one_cent": round(float((cents <= 1 + 1e-9).mean()), 4),
        "days_differing_by_half_cent": int((cents > 0.25).sum()),
        "max_return_gap_pp_excluding_distributions": round(float(return_gap.max()), 4),
        # Yahoo reports volume rounded to the nearest 100 shares.
        "volume_within_rounding_share": round(float((volume_gap <= 50).mean()), 4),
        "max_volume_difference_%": round(float(volume_gap_pct.max()), 3),
        "distribution_steps": steps,
    }
    return daily, summary


def run(
    external_dir: Path = EXTERNAL_DIR,
    raw_dir: Path = config.RAW_DIR,
    out_dir: Path = CROSSCHECK_DIR,
) -> dict:
    files = sorted((external_dir / "tmx").glob("*.csv"))
    if not files:
        return {}
    announced_path = external_dir / "xeqt_distributions.csv"
    announced = (
        pd.read_csv(announced_path, dtype={"Ex_Date": str})
        if announced_path.exists()
        else pd.DataFrame(columns=["Ex_Date", "Cash_Per_Unit", "Source"])
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    windows, frames = [], []
    for path in files:
        ticker = path.name.split("_")[0] + ".TO"
        yahoo = pd.read_csv(raw_dir / f"{ticker}.csv", parse_dates=["Date"])
        daily, summary = compare(yahoo, load_tmx(path))
        for step in summary["distribution_steps"]:
            match = announced[announced["Ex_Date"] == step["Ex_Date"]]
            step["Announced_Cash_Per_Unit"] = (
                float(match["Cash_Per_Unit"].iloc[0]) if len(match) else None
            )
            step["Source"] = match["Source"].iloc[0] if len(match) else None
        windows.append({"ticker": ticker, "file": path.name, **summary})
        frames.append(daily.assign(Ticker=ticker))

    pd.concat(frames).to_csv(out_dir / "daily_comparison.csv", index=False)
    result = {"source": "TMX Money price history (money.tmx.com)", "windows": windows}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result
