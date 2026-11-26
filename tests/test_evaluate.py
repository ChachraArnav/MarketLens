import numpy as np

from marketlens.evaluate import best_threshold, threshold_metrics


def test_threshold_metrics_counts():
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.05])
    m = threshold_metrics(y, p, 0.5)
    assert (
        m["True_Positives"],
        m["False_Positives"],
        m["False_Negatives"],
        m["True_Negatives"],
    ) == (1, 1, 1, 2)
    assert m["Precision"] == 0.5 and m["Recall"] == 0.5


def test_best_threshold_picks_highest_f1():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.6, 0.55, 0.3, 0.2, 0.1, 0.1])
    assert best_threshold(y, p, thresholds=[0.1, 0.5, 0.9]) == 0.5


def test_no_predictions_gives_zero_not_error():
    m = threshold_metrics(np.array([1, 0]), np.array([0.1, 0.1]), 0.9)
    assert m["Precision"] == 0 and m["Recall"] == 0 and m["F1"] == 0
