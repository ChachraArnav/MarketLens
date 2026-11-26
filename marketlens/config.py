"""Every research decision in one place, so results are reproducible and easy to audit."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

PRICES_FILE = PROCESSED_DIR / "all_etfs.csv"
DATASET_FILE = PROCESSED_DIR / "final_ml_dataset.csv"

# Canadian-listed ETFs covering broad equity, sectors, bonds and gold.
TICKERS = [
    "VFV.TO",
    "XEQT.TO",
    "ZNQ.TO",
    "XIC.TO",
    "XEF.TO",
    "XEC.TO",
    "XFN.TO",
    "XEG.TO",
    "XBB.TO",
    "XGD.TO",
    "XRE.TO",
    "XIT.TO",
    "CGL.TO",
]
START_DATE = "2021-01-01"

# Model inputs. Every one uses information available at the close of day t.
FEATURES = [
    "Return_1D",
    "Return_5D",
    "Volatility_20D",
    "Price_vs_MA10",
    "Volume_Change",
]

# Labels look 5 trading days ahead.
HORIZON = 5
FIXED_DROP = -0.03  # Fixed_Risk: the next 5 days lose at least 3%
ADJUSTED_MULTIPLIER = 1.5  # Adjusted_Risk: a drop beyond 1.5 sigma of recent volatility
TARGETS = ["Fixed_Risk", "Adjusted_Risk"]

# Chronological periods. The 2026 test period is used exactly once, after
# the model and threshold have been chosen on 2025.
TRAIN_END = "2024-12-31"
VALIDATION_START = "2025-01-01"
VALIDATION_END = "2025-12-31"
TEST_START = "2026-01-01"

RANDOM_STATE = 42
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ... 0.95
