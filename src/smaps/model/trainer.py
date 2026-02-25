"""Baseline model trainer using Logistic Regression with StandardScaler."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from smaps.models import Direction


@dataclass
class TrainedModel:
    """A trained ML model wrapping a scikit-learn pipeline."""

    pipeline: Pipeline
    feature_names: list[str]
    metrics: dict[str, float] = field(default_factory=dict)

    def predict(self, features: dict[str, float]) -> tuple[Direction, float]:
        """Predict direction and confidence for a single feature vector.

        Returns:
            Tuple of (Direction, confidence) where confidence is 0.0–1.0.
        """
        X = pd.DataFrame([features])[self.feature_names]
        X = X.fillna(0.0)
        proba = self.pipeline.predict_proba(X)[0]
        classes = list(self.pipeline.classes_)

        if 1 in classes:
            up_idx = classes.index(1)
            up_prob = float(proba[up_idx])
        else:
            up_prob = 0.0

        if up_prob >= 0.5:
            return Direction.UP, up_prob
        else:
            return Direction.DOWN, 1.0 - up_prob


def train_model(
    features_df: pd.DataFrame,
    labels: pd.Series,  # type: ignore[type-arg]
    test_ratio: float = 0.2,
) -> TrainedModel:
    """Train a LogisticRegression model with time-based train/test split.

    Args:
        features_df: Feature matrix (rows=samples, columns=features),
            ordered chronologically.
        labels: Binary labels (1=UP, 0=DOWN), aligned with features_df.
        test_ratio: Fraction of data held out for testing (from the end).

    Returns:
        TrainedModel with trained pipeline and test-set metrics.
    """
    # Replace NaN with 0.0 for training
    features_df = features_df.fillna(0.0)

    # Time-based split (no shuffle)
    n = len(features_df)
    split_idx = int(n * (1.0 - test_ratio))
    split_idx = max(1, min(split_idx, n - 1))

    X_train = features_df.iloc[:split_idx]
    X_test = features_df.iloc[split_idx:]
    y_train = labels.iloc[:split_idx]
    y_test = labels.iloc[split_idx:]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42, max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)

    accuracy = float(pipeline.score(X_test, y_test))

    return TrainedModel(
        pipeline=pipeline,
        feature_names=list(features_df.columns),
        metrics={
            "accuracy": accuracy,
            "train_size": float(len(X_train)),
            "test_size": float(len(X_test)),
        },
    )
