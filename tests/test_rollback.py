"""Tests for rollback on regression (US-504)."""

from __future__ import annotations

import datetime
import logging

import pytest

from smaps.db import (
    ensure_schema,
    get_connection,
    upsert_bars,
)
from smaps.model.registry import load_latest_model, save_model
from smaps.model.trainer import train_model
from smaps.models import OHLCVBar
from smaps.retrainer import retrain, retrain_with_validation


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
    days: int = 100,
) -> None:
    """Insert synthetic daily OHLCV bars with alternating UP/DOWN pattern."""
    bars = []
    for i in range(days):
        dt = start + datetime.timedelta(days=i)
        # Alternating close: 100, 110, 100, 110, ...
        close = 100.0 + (i % 2) * 10.0
        bars.append(_make_bar(ticker=ticker, date=dt, close=close))
    upsert_bars(conn, bars)


class TestRollback:
    """Tests that rollback keeps previous model active on OOS gate failure."""

    def test_previous_model_remains_active_after_rollback(self, tmp_path) -> None:
        """If new model fails OOS gate, load_latest_model still returns the previous model."""
        conn = _setup_db()
        _insert_price_history(conn, days=100)
        models_dir = str(tmp_path / "models")

        # Train an initial model via retrain() (always saves)
        initial_record = retrain(conn, "AAPL", models_dir=models_dir)
        assert initial_record.version == 1

        # Load the current model — this is the model that should survive rollback
        result_before = load_latest_model(conn, "AAPL")
        assert result_before is not None
        _, record_before = result_before
        assert record_before.version == 1

        # Now retrain_with_validation — the new model trains on the
        # exact same data, so OOS accuracy will be equal to current model.
        # Equal accuracy → strict > comparison → gate blocks → rollback.
        result = retrain_with_validation(
            conn, "AAPL", models_dir=models_dir, oos_days=30,
        )

        # retrain_with_validation returns None on rollback
        assert result is None

        # Previous model is still the latest (version 1)
        result_after = load_latest_model(conn, "AAPL")
        assert result_after is not None
        _, record_after = result_after
        assert record_after.version == 1

    def test_returns_none_on_rollback(self, tmp_path) -> None:
        """retrain_with_validation returns None when OOS gate blocks."""
        conn = _setup_db()
        _insert_price_history(conn, days=100)
        models_dir = str(tmp_path / "models")

        # Seed a model so the OOS gate has a comparison
        retrain(conn, "AAPL", models_dir=models_dir)

        # Same data → same model → equal accuracy → blocked
        result = retrain_with_validation(
            conn, "AAPL", models_dir=models_dir, oos_days=30,
        )

        assert result is None

    def test_model_version_not_incremented_on_rollback(self, tmp_path) -> None:
        """Rollback does not increment the model version in the registry."""
        conn = _setup_db()
        _insert_price_history(conn, days=100)
        models_dir = str(tmp_path / "models")

        retrain(conn, "AAPL", models_dir=models_dir)  # v1

        # Attempt retrain that gets rolled back
        retrain_with_validation(conn, "AAPL", models_dir=models_dir, oos_days=30)

        # Version should still be 1
        result = load_latest_model(conn, "AAPL")
        assert result is not None
        _, record = result
        assert record.version == 1

    def test_promotes_when_no_current_model(self, tmp_path) -> None:
        """First model for a ticker is always promoted (no current to compare)."""
        conn = _setup_db()
        _insert_price_history(conn, days=100)
        models_dir = str(tmp_path / "models")

        result = retrain_with_validation(
            conn, "AAPL", models_dir=models_dir, oos_days=30,
        )

        # Should be promoted (no existing model to beat)
        assert result is not None
        assert result.version == 1
        assert result.ticker == "AAPL"

    def test_promotes_model_record_returned(self, tmp_path) -> None:
        """When promoted, returns a valid ModelRecord."""
        conn = _setup_db()
        _insert_price_history(conn, days=100)
        models_dir = str(tmp_path / "models")

        result = retrain_with_validation(
            conn, "AAPL", models_dir=models_dir, oos_days=30,
        )

        assert result is not None
        assert result.ticker == "AAPL"
        assert result.version >= 1

    def test_rollback_does_not_create_artifact_file(self, tmp_path) -> None:
        """On rollback, no new joblib file is created for the blocked model."""
        conn = _setup_db()
        _insert_price_history(conn, days=100)
        models_dir = tmp_path / "models"

        retrain(conn, "AAPL", models_dir=str(models_dir))  # v1
        files_before = set(models_dir.iterdir())

        # Rollback — should not create a new artifact
        retrain_with_validation(
            conn, "AAPL", models_dir=str(models_dir), oos_days=30,
        )

        files_after = set(models_dir.iterdir())
        assert files_before == files_after

    def test_raises_on_insufficient_data(self, tmp_path) -> None:
        """retrain_with_validation raises ValueError with fewer than 2 trading days."""
        conn = _setup_db()
        upsert_bars(conn, [_make_bar(date=datetime.date(2025, 1, 1))])

        with pytest.raises(ValueError, match="Insufficient data"):
            retrain_with_validation(conn, "AAPL", models_dir=str(tmp_path / "models"))


class TestRollbackLogging:
    """Tests that rollback events are logged with reason and metrics."""

    def test_logs_rollback_event(self, tmp_path, caplog: pytest.LogCaptureFixture) -> None:
        """Rollback logs a WARNING with reason and metrics."""
        conn = _setup_db()
        _insert_price_history(conn, days=100)
        models_dir = str(tmp_path / "models")

        # Seed a model to force OOS comparison
        retrain(conn, "AAPL", models_dir=models_dir)

        with caplog.at_level(logging.WARNING, logger="smaps.retrainer"):
            retrain_with_validation(conn, "AAPL", models_dir=models_dir, oos_days=30)

        assert any("rollback" in msg for msg in caplog.messages)
        assert any("oos_gate_failed" in msg for msg in caplog.messages)
        assert any("AAPL" in msg for msg in caplog.messages)

    def test_rollback_log_includes_metrics(self, tmp_path, caplog: pytest.LogCaptureFixture) -> None:
        """Rollback log includes OOS accuracy metrics."""
        conn = _setup_db()
        _insert_price_history(conn, days=100)
        models_dir = str(tmp_path / "models")

        retrain(conn, "AAPL", models_dir=models_dir)

        with caplog.at_level(logging.WARNING, logger="smaps.retrainer"):
            retrain_with_validation(conn, "AAPL", models_dir=models_dir, oos_days=30)

        rollback_msgs = [msg for msg in caplog.messages if "rollback" in msg]
        assert len(rollback_msgs) >= 1
        msg = rollback_msgs[0]
        assert "new_oos_accuracy=" in msg
        assert "current_oos_accuracy=" in msg

    def test_logs_retrain_complete_on_promote(self, tmp_path, caplog: pytest.LogCaptureFixture) -> None:
        """When promoted, logs retrain_complete (not rollback)."""
        conn = _setup_db()
        _insert_price_history(conn, days=100)
        models_dir = str(tmp_path / "models")

        with caplog.at_level(logging.INFO, logger="smaps.retrainer"):
            retrain_with_validation(conn, "AAPL", models_dir=models_dir, oos_days=30)

        assert any("retrain_complete" in msg for msg in caplog.messages)
        # No rollback message
        rollback_msgs = [msg for msg in caplog.messages if "rollback" in msg]
        assert len(rollback_msgs) == 0
