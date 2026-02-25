"""Tests for US-304: daily prediction function."""

from __future__ import annotations

import datetime
import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from smaps.db import ensure_schema, get_connection
from smaps.model.predictor import predict
from smaps.model.registry import save_model
from smaps.model.trainer import TrainedModel, train_model
from smaps.models import Direction, PredictionResult


@pytest.fixture()
def conn(tmp_path: object) -> sqlite3.Connection:
    """In-memory DB with schema applied."""
    c = get_connection(":memory:")
    ensure_schema(c)
    return c


@pytest.fixture()
def models_dir(tmp_path):  # type: ignore[no-untyped-def]
    """Temporary directory for model artifacts."""
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture()
def trained_model() -> TrainedModel:
    """A simple trained model for testing."""
    rng = np.random.RandomState(42)
    n = 50
    features_df = pd.DataFrame({
        "return_1d": rng.randn(n),
        "return_5d": rng.randn(n),
        "return_10d": rng.randn(n),
        "ma_ratio_5_20": rng.randn(n),
        "volume_change_1d": rng.randn(n),
        "volatility_20d": np.abs(rng.randn(n)),
        "rsi_14": rng.uniform(0, 100, n),
        "latest_sentiment_score": rng.uniform(-1, 1, n),
        "sentiment_ma_5d": rng.uniform(-1, 1, n),
        "pe_ratio": rng.uniform(5, 50, n),
        "eps": rng.uniform(0.5, 10, n),
        "market_cap": rng.uniform(1e9, 1e12, n),
    })
    labels = pd.Series(rng.randint(0, 2, size=n))
    return train_model(features_df, labels)


class TestPredict:
    """Tests for the predict() function."""

    def test_returns_prediction_result(
        self, conn: sqlite3.Connection, models_dir, trained_model: TrainedModel  # type: ignore[no-untyped-def]
    ) -> None:
        save_model(conn, "AAPL", trained_model, models_dir)
        result = predict(conn, "AAPL", datetime.date(2025, 1, 15))
        assert isinstance(result, PredictionResult)

    def test_output_schema(
        self, conn: sqlite3.Connection, models_dir, trained_model: TrainedModel  # type: ignore[no-untyped-def]
    ) -> None:
        save_model(conn, "AAPL", trained_model, models_dir)
        result = predict(conn, "AAPL", datetime.date(2025, 1, 15))
        assert result.ticker == "AAPL"
        assert result.prediction_date == datetime.date(2025, 1, 15)
        assert isinstance(result.direction, Direction)
        assert 0.0 <= result.confidence <= 1.0
        assert result.model_version == "v1"

    def test_direction_is_up_or_down(
        self, conn: sqlite3.Connection, models_dir, trained_model: TrainedModel  # type: ignore[no-untyped-def]
    ) -> None:
        save_model(conn, "AAPL", trained_model, models_dir)
        result = predict(conn, "AAPL", datetime.date(2025, 1, 15))
        assert result.direction in (Direction.UP, Direction.DOWN)

    def test_model_version_from_registry(
        self, conn: sqlite3.Connection, models_dir, trained_model: TrainedModel  # type: ignore[no-untyped-def]
    ) -> None:
        save_model(conn, "AAPL", trained_model, models_dir)
        save_model(conn, "AAPL", trained_model, models_dir)
        result = predict(conn, "AAPL", datetime.date(2025, 1, 15))
        assert result.model_version == "v2"

    def test_raises_when_no_model(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(RuntimeError, match="No trained model found"):
            predict(conn, "AAPL", datetime.date(2025, 1, 15))

    def test_raises_for_unknown_ticker(
        self, conn: sqlite3.Connection, models_dir, trained_model: TrainedModel  # type: ignore[no-untyped-def]
    ) -> None:
        save_model(conn, "AAPL", trained_model, models_dir)
        with pytest.raises(RuntimeError, match="No trained model found"):
            predict(conn, "MSFT", datetime.date(2025, 1, 15))

    def test_uses_build_features(
        self, conn: sqlite3.Connection, models_dir, trained_model: TrainedModel  # type: ignore[no-untyped-def]
    ) -> None:
        """Verify that predict() calls build_features with correct args."""
        save_model(conn, "AAPL", trained_model, models_dir)
        date = datetime.date(2025, 1, 15)
        with patch("smaps.model.predictor.build_features") as mock_bf:
            mock_bf.return_value = {name: 0.0 for name in trained_model.feature_names}
            predict(conn, "AAPL", date)
            mock_bf.assert_called_once_with(conn, "AAPL", date)

    def test_with_mock_model(self, conn: sqlite3.Connection) -> None:
        """Test with a fully mocked model to verify output schema."""
        mock_model = MagicMock(spec=TrainedModel)
        mock_model.predict.return_value = (Direction.UP, 0.75)

        mock_record = MagicMock()
        mock_record.version = 3

        with patch("smaps.model.predictor.load_latest_model") as mock_load, \
             patch("smaps.model.predictor.build_features") as mock_bf:
            mock_load.return_value = (mock_model, mock_record)
            mock_bf.return_value = {"f1": 1.0, "f2": 2.0}

            result = predict(conn, "GOOG", datetime.date(2025, 6, 1))

        assert isinstance(result, PredictionResult)
        assert result.ticker == "GOOG"
        assert result.prediction_date == datetime.date(2025, 6, 1)
        assert result.direction == Direction.UP
        assert result.confidence == 0.75
        assert result.model_version == "v3"
