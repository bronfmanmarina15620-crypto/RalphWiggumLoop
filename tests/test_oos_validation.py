"""Tests for out-of-sample validation gate (US-503)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from smaps.model.trainer import train_model
from smaps.retrainer import validate_oos


def _make_separable_data(
    n: int = 100, seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create a linearly separable dataset.

    Feature f1 ranges from -1 to 1; label is UP (1) when f1 > 0.
    """
    rng = np.random.RandomState(seed)
    f1 = np.linspace(-1, 1, n)
    f2 = rng.randn(n) * 0.01  # near-zero noise feature
    features = pd.DataFrame({"f1": f1, "f2": f2})
    labels = pd.Series((f1 > 0).astype(int))
    return features, labels


def _make_random_labels(n: int = 100, seed: int = 99) -> pd.Series:
    """Create random binary labels."""
    rng = np.random.RandomState(seed)
    return pd.Series(rng.randint(0, 2, size=n))


class TestValidateOos:
    """Tests for the validate_oos function."""

    def test_promotes_when_no_current_model(self) -> None:
        """No current model -> always promote."""
        features, labels = _make_separable_data()
        model = train_model(features, labels)

        promote, metrics = validate_oos(
            model, features, labels, current_model=None,
        )

        assert promote is True
        assert metrics["promoted"] == 1.0
        assert "new_oos_accuracy" in metrics

    def test_promotes_when_new_model_better(self) -> None:
        """New model with higher OOS accuracy -> promoted."""
        features, labels = _make_separable_data()
        good_model = train_model(features, labels)
        bad_model = train_model(features, _make_random_labels(n=100))

        promote, metrics = validate_oos(
            new_model=good_model,
            features_df=features,
            labels=labels,
            current_model=bad_model,
        )

        assert promote is True
        assert metrics["new_oos_accuracy"] > metrics["current_oos_accuracy"]
        assert metrics["promoted"] == 1.0

    def test_blocks_when_new_model_worse(self) -> None:
        """New model with lower OOS accuracy -> blocked."""
        features, labels = _make_separable_data()
        good_model = train_model(features, labels)
        bad_model = train_model(features, _make_random_labels(n=100))

        promote, metrics = validate_oos(
            new_model=bad_model,
            features_df=features,
            labels=labels,
            current_model=good_model,
        )

        assert promote is False
        assert metrics["promoted"] == 0.0

    def test_blocks_when_equal_accuracy(self) -> None:
        """Same model as both new and current -> equal accuracy -> blocked."""
        features, labels = _make_separable_data()
        model = train_model(features, labels)

        promote, metrics = validate_oos(
            new_model=model,
            features_df=features,
            labels=labels,
            current_model=model,
        )

        assert promote is False
        assert metrics["promoted"] == 0.0
        assert metrics["new_oos_accuracy"] == metrics["current_oos_accuracy"]

    def test_promotes_with_insufficient_data(self) -> None:
        """When samples <= oos_days, promote by default."""
        features, labels = _make_separable_data(n=10)
        model = train_model(features, labels)

        promote, metrics = validate_oos(
            model, features, labels, oos_days=30,
        )

        assert promote is True
        assert metrics["promoted"] == 1.0

    def test_oos_size_matches_oos_days(self) -> None:
        """OOS hold-out size matches oos_days parameter."""
        features, labels = _make_separable_data(n=100)
        model = train_model(features, labels)

        _, metrics = validate_oos(model, features, labels, oos_days=20)

        assert metrics["oos_size"] == 20.0

    def test_default_oos_days_is_30(self) -> None:
        """Default oos_days=30 produces 30-sample hold-out."""
        features, labels = _make_separable_data(n=100)
        model = train_model(features, labels)

        _, metrics = validate_oos(model, features, labels)

        assert metrics["oos_size"] == 30.0

    def test_metrics_contain_current_accuracy_when_current_provided(self) -> None:
        """Metrics include current_oos_accuracy when a current model is given."""
        features, labels = _make_separable_data()
        model = train_model(features, labels)

        _, metrics = validate_oos(
            model, features, labels, current_model=model,
        )

        assert "current_oos_accuracy" in metrics
        assert 0.0 <= metrics["current_oos_accuracy"] <= 1.0

    def test_metrics_exclude_current_accuracy_when_no_current(self) -> None:
        """Metrics do not contain current_oos_accuracy when no current model."""
        features, labels = _make_separable_data()
        model = train_model(features, labels)

        _, metrics = validate_oos(model, features, labels, current_model=None)

        assert "current_oos_accuracy" not in metrics

    def test_new_oos_accuracy_in_valid_range(self) -> None:
        """new_oos_accuracy is between 0.0 and 1.0."""
        features, labels = _make_separable_data()
        model = train_model(features, labels)

        _, metrics = validate_oos(model, features, labels)

        assert 0.0 <= metrics["new_oos_accuracy"] <= 1.0


class TestValidateOosLogging:
    """Tests for validate_oos structured logging."""

    def test_logs_promote_no_current(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Promote decision with no current model is logged."""
        features, labels = _make_separable_data()
        model = train_model(features, labels)

        with caplog.at_level(logging.INFO, logger="smaps.retrainer"):
            validate_oos(model, features, labels, current_model=None)

        assert any("oos_gate" in msg for msg in caplog.messages)
        assert any("promote" in msg for msg in caplog.messages)
        assert any("no_current_model" in msg for msg in caplog.messages)

    def test_logs_promote_better_model(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Promote decision with better model is logged."""
        features, labels = _make_separable_data()
        good_model = train_model(features, labels)
        bad_model = train_model(features, _make_random_labels(n=100))

        with caplog.at_level(logging.INFO, logger="smaps.retrainer"):
            validate_oos(good_model, features, labels, current_model=bad_model)

        assert any("oos_gate" in msg for msg in caplog.messages)
        assert any("result=promote" in msg for msg in caplog.messages)

    def test_logs_blocked(self, caplog: pytest.LogCaptureFixture) -> None:
        """Blocked decision is logged with metrics."""
        features, labels = _make_separable_data()
        good_model = train_model(features, labels)
        bad_model = train_model(features, _make_random_labels(n=100))

        with caplog.at_level(logging.INFO, logger="smaps.retrainer"):
            validate_oos(bad_model, features, labels, current_model=good_model)

        assert any("oos_gate" in msg for msg in caplog.messages)
        assert any("blocked" in msg for msg in caplog.messages)

    def test_logs_insufficient_data(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Insufficient data decision is logged."""
        features, labels = _make_separable_data(n=10)
        model = train_model(features, labels)

        with caplog.at_level(logging.INFO, logger="smaps.retrainer"):
            validate_oos(model, features, labels, oos_days=30)

        assert any("oos_gate" in msg for msg in caplog.messages)
        assert any("insufficient_data" in msg for msg in caplog.messages)
