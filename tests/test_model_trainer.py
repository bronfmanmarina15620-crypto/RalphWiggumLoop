"""Tests for US-302: Train baseline model (Logistic Regression)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from smaps.model.trainer import TrainedModel, train_model
from smaps.models import Direction


def _make_synthetic_data(
    n: int = 100, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:  # type: ignore[type-arg]
    """Create synthetic features and labels for testing.

    Labels correlate with feature_a so the model can learn a pattern.
    """
    rng = np.random.RandomState(seed)
    features = pd.DataFrame(
        {
            "feature_a": rng.randn(n),
            "feature_b": rng.randn(n),
            "feature_c": rng.randn(n),
        }
    )
    labels = pd.Series((features["feature_a"] > 0).astype(int))
    return features, labels


class TestTrainModel:
    """Tests for the train_model function."""

    def test_returns_trained_model(self) -> None:
        features, labels = _make_synthetic_data()
        model = train_model(features, labels)
        assert isinstance(model, TrainedModel)

    def test_model_produces_predictions(self) -> None:
        features, labels = _make_synthetic_data()
        model = train_model(features, labels)
        sample = {"feature_a": 1.0, "feature_b": 0.5, "feature_c": -0.3}
        direction, confidence = model.predict(sample)
        assert isinstance(direction, Direction)
        assert direction in (Direction.UP, Direction.DOWN)
        assert 0.0 <= confidence <= 1.0

    def test_time_based_split_no_shuffle(self) -> None:
        features, labels = _make_synthetic_data()
        model = train_model(features, labels, test_ratio=0.2)
        assert model.metrics["train_size"] == 80.0
        assert model.metrics["test_size"] == 20.0

    def test_metrics_include_accuracy(self) -> None:
        features, labels = _make_synthetic_data()
        model = train_model(features, labels)
        assert "accuracy" in model.metrics
        assert 0.0 <= model.metrics["accuracy"] <= 1.0

    def test_feature_names_preserved(self) -> None:
        features, labels = _make_synthetic_data()
        model = train_model(features, labels)
        assert model.feature_names == ["feature_a", "feature_b", "feature_c"]

    def test_handles_nan_in_training_data(self) -> None:
        features, labels = _make_synthetic_data(50)
        features.iloc[0, 0] = float("nan")
        model = train_model(features, labels)
        assert isinstance(model, TrainedModel)

    def test_handles_nan_in_prediction(self) -> None:
        features, labels = _make_synthetic_data()
        model = train_model(features, labels)
        sample = {"feature_a": float("nan"), "feature_b": 0.5, "feature_c": -0.3}
        direction, confidence = model.predict(sample)
        assert isinstance(direction, Direction)
        assert 0.0 <= confidence <= 1.0

    def test_uses_logistic_regression(self) -> None:
        features, labels = _make_synthetic_data()
        model = train_model(features, labels)
        clf = model.pipeline.named_steps["clf"]
        assert isinstance(clf, LogisticRegression)

    def test_uses_standard_scaler(self) -> None:
        features, labels = _make_synthetic_data()
        model = train_model(features, labels)
        scaler = model.pipeline.named_steps["scaler"]
        assert isinstance(scaler, StandardScaler)
