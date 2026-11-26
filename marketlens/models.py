"""The model families compared in the research."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

MODEL_NAMES = ["Logistic Regression", "Random Forest", "Neural Network"]


def make_model(name: str):
    """Return an unfitted model.

    Scaling sits inside the pipeline, so it is fit on training data only.
    """
    if name == "Baseline":
        return DummyClassifier(strategy="most_frequent")
    if name == "Logistic Regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
            ),
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=10,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "Neural Network":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=config.RANDOM_STATE,
            ),
        )
    raise ValueError(f"Unknown model: {name}")


def is_pipeline(model) -> bool:
    return isinstance(model, Pipeline)
