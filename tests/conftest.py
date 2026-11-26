import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices() -> pd.DataFrame:
    """Two synthetic ETFs, 300 business days, deterministic random walks."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=300)
    frames = []
    for ticker, drift in [("AAA.TO", 0.0005), ("BBB.TO", -0.0002)]:
        close = 50 * np.exp(np.cumsum(rng.normal(drift, 0.012, len(dates))))
        volume = rng.integers(1_000, 50_000, len(dates))
        frames.append(
            pd.DataFrame({"Date": dates, "Close": close, "Volume": volume, "Ticker": ticker})
        )
    return pd.concat(frames, ignore_index=True)
