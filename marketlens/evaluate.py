"""Model comparison on 2025, then a single, pre-registered evaluation on 2026.

Protocol (fixed before looking at 2026):
  1. Train every model on 2021-2024 (purged).
  2. On 2025 validation, pick the model with the highest average precision,
     then the threshold with the highest F1 for that model.
  3. Freeze both choices and score them once on the 2026 test period.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from . import config
from .models import MODEL_NAMES, make_model
from .splits import Splits


def threshold_metrics(y_true, probabilities, threshold: float) -> dict:
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, predicted),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "True_Positives": int(tp),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Negatives": int(tn),
    }


def best_threshold(y_true, probabilities, thresholds=config.THRESHOLDS) -> float:
    """Threshold with the highest F1; ties go to the higher (more conservative) threshold."""
    scores = [(threshold_metrics(y_true, probabilities, t)["F1"], t) for t in thresholds]
    return max(scores)[1]


@dataclass
class HoldoutResult:
    risk_definition: str
    model: str
    threshold: float
    validation_average_precision: float
    validation_base_rate: float
    test_average_precision: float
    test_base_rate: float
    test_lift: float
    test_precision: float
    test_recall: float
    test_f1: float
    test_observations: int
    test_events: int
    test_start: str
    test_end: str


def run(splits: Splits, results_dir=config.RESULTS_DIR) -> dict:
    research_dir = results_dir / "ml_research"
    holdout_dir = results_dir / "holdout"
    research_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    X_train = splits.train[config.FEATURES]
    X_val = splits.validation[config.FEATURES]
    X_test = splits.test[config.FEATURES]

    comparison, sweeps, holdout = [], [], []

    for target in config.TARGETS:
        y_train = splits.train[target]
        y_val = splits.validation[target]
        y_test = splits.test[target]
        fitted, val_ap = {}, {}

        for name in ["Baseline", *MODEL_NAMES]:
            model = make_model(name).fit(X_train, y_train)
            probabilities = model.predict_proba(X_val)[:, 1]
            metrics = threshold_metrics(y_val, probabilities, 0.5)
            ap = average_precision_score(y_val, probabilities)
            comparison.append(
                {
                    "Risk_Definition": target,
                    "Model": name,
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1": metrics["F1"],
                    "Average_Precision": ap,
                    "Baseline_AP": float(y_val.mean()),
                }
            )
            if name == "Baseline":
                continue
            fitted[name], val_ap[name] = model, ap

            model_id = f"{target}_{name.replace(' ', '_')}"
            pd.DataFrame(
                {
                    "Date": splits.validation["Date"].values,
                    "Ticker": splits.validation["Ticker"].values,
                    "Actual": y_val.values,
                    "Probability": probabilities,
                }
            ).to_csv(research_dir / f"{model_id}_predictions.csv", index=False)
            precision, recall, _ = precision_recall_curve(y_val, probabilities)
            pd.DataFrame({"Precision": precision, "Recall": recall}).to_csv(
                research_dir / f"{model_id}_curve.csv", index=False
            )
            for t in config.THRESHOLDS:
                sweeps.append(
                    {
                        "Risk_Definition": target,
                        "Model": name,
                        **threshold_metrics(y_val, probabilities, t),
                    }
                )

        # Step 2: choose on validation only.
        chosen = max(val_ap, key=val_ap.get)
        model = fitted[chosen]
        val_probabilities = model.predict_proba(X_val)[:, 1]
        threshold = best_threshold(y_val, val_probabilities)

        # Step 3: one look at the test period.
        test_probabilities = model.predict_proba(X_test)[:, 1]
        test_ap = average_precision_score(y_test, test_probabilities)
        test_metrics = threshold_metrics(y_test, test_probabilities, threshold)
        base_rate = float(y_test.mean())
        holdout.append(
            HoldoutResult(
                risk_definition=target,
                model=chosen,
                threshold=threshold,
                validation_average_precision=float(val_ap[chosen]),
                validation_base_rate=float(y_val.mean()),
                test_average_precision=float(test_ap),
                test_base_rate=base_rate,
                test_lift=float(test_ap / base_rate) if base_rate else float("nan"),
                test_precision=float(test_metrics["Precision"]),
                test_recall=float(test_metrics["Recall"]),
                test_f1=float(test_metrics["F1"]),
                test_observations=int(len(y_test)),
                test_events=int(y_test.sum()),
                test_start=str(pd.to_datetime(splits.test["Date"]).min().date()),
                test_end=str(pd.to_datetime(splits.test["Date"]).max().date()),
            )
        )
        pd.DataFrame(
            {
                "Date": splits.test["Date"].values,
                "Ticker": splits.test["Ticker"].values,
                "Actual": y_test.values,
                "Probability": test_probabilities,
            }
        ).to_csv(holdout_dir / f"{target}_test_predictions.csv", index=False)
        precision, recall, _ = precision_recall_curve(y_test, test_probabilities)
        pd.DataFrame({"Precision": precision, "Recall": recall}).to_csv(
            holdout_dir / f"{target}_test_curve.csv", index=False
        )

        if chosen == "Random Forest":
            pd.DataFrame(
                {"Feature": config.FEATURES, "Importance": model.feature_importances_}
            ).sort_values("Importance", ascending=False).to_csv(
                holdout_dir / f"{target}_feature_importance.csv", index=False
            )

    pd.DataFrame(comparison).to_csv(results_dir / "final_model_comparison.csv", index=False)
    pd.DataFrame(sweeps).to_csv(results_dir / "threshold_comparison.csv", index=False)
    summary = {
        "protocol": __doc__.strip().split("\n\n", 1)[1],
        "periods": {
            "train": [
                str(pd.to_datetime(splits.train["Date"]).min().date()),
                str(pd.to_datetime(splits.train["Date"]).max().date()),
            ],
            "validation": [config.VALIDATION_START, config.VALIDATION_END],
            "test": [holdout[0].test_start, holdout[0].test_end] if holdout else None,
        },
        "results": [asdict(h) for h in holdout],
    }
    (holdout_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
