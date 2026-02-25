"""Tests for automated retraining pipeline (US-502)."""

from __future__ import annotations

import datetime
import logging

import pytest

from smaps.db import (
    ensure_schema,
    get_connection,
    upsert_bars,
)
from smaps.model.registry import load_latest_model
from smaps.models import OHLCVBar
from smaps.retrainer import retrain


def _setup_db():
    conn = get_connection(":memory:")
    ensure_schema(conn)
    return conn


def _make_bar(
    ticker: str = "AAPL",
    date: datetime.date | None = None,
    close: float = 100.0,
) -> OHLCVBar:
    return OHLCVBar(
        ticker=ticker,
        date=date or datetime.date(2025, 1, 15),
        open=close - 1.0,
        high=close + 1.0,
        low=close - 2.0,
        close=close,
        volume=1000,
    )


def _insert_price_history(
    conn,
    ticker: str = "AAPL",
    start: datetime.date = datetime.date(2025, 1, 1),
    days: int = 30,
) -> None:
    """Insert synthetic daily OHLCV bars with alternating UP/DOWN pattern."""
    bars = []
    for i in range(days):
        dt = start + datetime.timedelta(days=i)
        # Alternating close: 100, 110, 100, 110, ...
        close = 100.0 + (i % 2) * 10.0
        bars.append(_make_bar(ticker=ticker, date=dt, close=close))
    upsert_bars(conn, bars)


class TestRetrain:
    """Test the retrain() function."""

    def test_returns_model_record(self, tmp_path) -> None:
        """retrain() returns a ModelRecord on success."""
        conn = _setup_db()
        _insert_price_history(conn, days=30)

        record = retrain(conn, "AAPL", models_dir=str(tmp_path / "models"))

        assert record.ticker == "AAPL"
        assert record.version == 1

    def test_new_model_version_created(self, tmp_path) -> None:
        """retrain() creates a new model version that can be loaded."""
        conn = _setup_db()
        _insert_price_history(conn, days=30)

        record = retrain(conn, "AAPL", models_dir=str(tmp_path / "models"))

        result = load_latest_model(conn, "AAPL")
        assert result is not None
        model, loaded_record = result
        assert loaded_record.version == record.version
        assert loaded_record.ticker == "AAPL"

    def test_version_increments_on_second_retrain(self, tmp_path) -> None:
        """Second retrain() increments the model version."""
        conn = _setup_db()
        _insert_price_history(conn, days=30)
        models_dir = str(tmp_path / "models")

        record1 = retrain(conn, "AAPL", models_dir=models_dir)
        record2 = retrain(conn, "AAPL", models_dir=models_dir)

        assert record2.version == record1.version + 1

    def test_uses_all_historical_data(self, tmp_path) -> None:
        """retrain() uses all available data, not just a recent window."""
        conn = _setup_db()
        _insert_price_history(conn, days=50)
        models_dir = str(tmp_path / "models")

        record = retrain(conn, "AAPL", models_dir=models_dir)

        # Load model and check metrics — train_size + test_size should
        # account for all date pairs (days - 1 total samples)
        result = load_latest_model(conn, "AAPL")
        assert result is not None
        model, _ = result
        total_samples = model.metrics["train_size"] + model.metrics["test_size"]
        # 50 days → 49 feature/label pairs
        assert total_samples == 49

    def test_model_produces_predictions(self, tmp_path) -> None:
        """The retrained model can make predictions."""
        conn = _setup_db()
        _insert_price_history(conn, days=30)

        retrain(conn, "AAPL", models_dir=str(tmp_path / "models"))

        result = load_latest_model(conn, "AAPL")
        assert result is not None
        model, _ = result
        # Predict with dummy features
        direction, confidence = model.predict({
            "return_1d": 0.01,
            "return_5d": 0.05,
            "return_10d": 0.10,
            "ma_ratio_5_20": 1.05,
            "volume_change_1d": 0.1,
            "volatility_20d": 0.02,
            "rsi_14": 55.0,
            "latest_sentiment_score": 0.3,
            "sentiment_ma_5d": 0.2,
            "pe_ratio": 25.0,
            "eps": 5.0,
            "market_cap": 1e12,
        })
        assert direction.value in ("UP", "DOWN")
        assert 0.0 <= confidence <= 1.0

    def test_raises_on_insufficient_data(self, tmp_path) -> None:
        """retrain() raises ValueError with fewer than 2 trading days."""
        conn = _setup_db()
        # Only 1 bar
        upsert_bars(conn, [_make_bar(date=datetime.date(2025, 1, 1))])

        with pytest.raises(ValueError, match="Insufficient data"):
            retrain(conn, "AAPL", models_dir=str(tmp_path / "models"))

    def test_raises_on_no_data(self, tmp_path) -> None:
        """retrain() raises ValueError with no trading days."""
        conn = _setup_db()

        with pytest.raises(ValueError, match="Insufficient data"):
            retrain(conn, "AAPL", models_dir=str(tmp_path / "models"))

    def test_metrics_logged_in_record(self, tmp_path) -> None:
        """Model record contains training metrics as JSON."""
        conn = _setup_db()
        _insert_price_history(conn, days=30)

        record = retrain(conn, "AAPL", models_dir=str(tmp_path / "models"))

        import json
        metrics = json.loads(record.metrics_json)
        assert "accuracy" in metrics
        assert "train_size" in metrics
        assert "test_size" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_ticker_isolation(self, tmp_path) -> None:
        """retrain() for one ticker does not affect another."""
        conn = _setup_db()
        _insert_price_history(conn, ticker="AAPL", days=30)
        _insert_price_history(conn, ticker="MSFT", days=30)
        models_dir = str(tmp_path / "models")

        retrain(conn, "AAPL", models_dir=models_dir)

        # MSFT should have no model
        assert load_latest_model(conn, "MSFT") is None
        # AAPL should have a model
        assert load_latest_model(conn, "AAPL") is not None


class TestRetrainLogging:
    """Test that retrain() emits structured log events."""

    def test_logs_retrain_start(self, tmp_path, caplog: pytest.LogCaptureFixture) -> None:
        """retrain() logs start event with ticker and date range."""
        conn = _setup_db()
        _insert_price_history(conn, days=10)

        with caplog.at_level(logging.INFO, logger="smaps.retrainer"):
            retrain(conn, "AAPL", models_dir=str(tmp_path / "models"))

        assert any("retrain_start" in msg for msg in caplog.messages)
        assert any("AAPL" in msg for msg in caplog.messages)

    def test_logs_retrain_complete(self, tmp_path, caplog: pytest.LogCaptureFixture) -> None:
        """retrain() logs completion event with version and accuracy."""
        conn = _setup_db()
        _insert_price_history(conn, days=10)

        with caplog.at_level(logging.INFO, logger="smaps.retrainer"):
            retrain(conn, "AAPL", models_dir=str(tmp_path / "models"))

        assert any("retrain_complete" in msg for msg in caplog.messages)
        assert any("version=" in msg for msg in caplog.messages)
        assert any("accuracy=" in msg for msg in caplog.messages)

    def test_no_utilization_gap_warning_when_all_data_used(
        self, tmp_path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """retrain() should NOT log data_utilization_gap when all data is used."""
        conn = _setup_db()
        _insert_price_history(conn, days=10)

        with caplog.at_level(logging.WARNING, logger="smaps.retrainer"):
            retrain(conn, "AAPL", models_dir=str(tmp_path / "models"))

        assert not any("data_utilization_gap" in msg for msg in caplog.messages)
