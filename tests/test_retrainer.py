"""Tests for retrain trigger based on accuracy degradation (US-501)."""

from __future__ import annotations

import datetime
import logging

import pytest

from smaps.db import (
    ensure_schema,
    get_connection,
    save_prediction,
    upsert_bars,
)
from smaps.models import Direction, OHLCVBar, PredictionResult
from smaps.retrainer import should_retrain


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


def _make_prediction(
    ticker: str = "AAPL",
    date: datetime.date | None = None,
    direction: Direction = Direction.UP,
) -> PredictionResult:
    return PredictionResult(
        ticker=ticker,
        prediction_date=date or datetime.date(2025, 1, 15),
        direction=direction,
        confidence=0.75,
        model_version="v1",
    )


class TestShouldRetrain:
    """Test retrain trigger logic."""

    def test_returns_false_when_no_predictions(self) -> None:
        """No predictions in window -> False (no evidence of degradation)."""
        conn = _setup_db()
        result = should_retrain(
            conn, "AAPL", threshold=0.50, window_days=30,
            as_of_date=datetime.date(2025, 2, 1),
        )
        assert result is False

    def test_returns_true_when_accuracy_below_threshold(self) -> None:
        """All wrong predictions -> accuracy 0.0 < 0.50 -> True."""
        conn = _setup_db()
        # Two predictions that are both wrong (predict UP, actual DOWN)
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 6), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 7), close=90.0),
            _make_bar(date=datetime.date(2025, 1, 8), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 9), close=85.0),
        ])
        save_prediction(conn, _make_prediction(
            direction=Direction.UP, date=datetime.date(2025, 1, 6),
        ))
        save_prediction(conn, _make_prediction(
            direction=Direction.UP, date=datetime.date(2025, 1, 8),
        ))

        result = should_retrain(
            conn, "AAPL", threshold=0.50, window_days=30,
            as_of_date=datetime.date(2025, 2, 1),
        )
        assert result is True

    def test_returns_false_when_accuracy_above_threshold(self) -> None:
        """All correct predictions -> accuracy 1.0 >= 0.50 -> False."""
        conn = _setup_db()
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 6), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 7), close=110.0),
            _make_bar(date=datetime.date(2025, 1, 8), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 9), close=115.0),
        ])
        save_prediction(conn, _make_prediction(
            direction=Direction.UP, date=datetime.date(2025, 1, 6),
        ))
        save_prediction(conn, _make_prediction(
            direction=Direction.UP, date=datetime.date(2025, 1, 8),
        ))

        result = should_retrain(
            conn, "AAPL", threshold=0.50, window_days=30,
            as_of_date=datetime.date(2025, 2, 1),
        )
        assert result is False

    def test_returns_false_when_accuracy_equals_threshold(self) -> None:
        """Accuracy exactly at threshold -> False (not below)."""
        conn = _setup_db()
        # 1 correct, 1 wrong -> accuracy = 0.50
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 6), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 7), close=110.0),  # UP (correct)
            _make_bar(date=datetime.date(2025, 1, 8), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 9), close=90.0),   # DOWN (wrong)
        ])
        save_prediction(conn, _make_prediction(
            direction=Direction.UP, date=datetime.date(2025, 1, 6),
        ))
        save_prediction(conn, _make_prediction(
            direction=Direction.UP, date=datetime.date(2025, 1, 8),
        ))

        result = should_retrain(
            conn, "AAPL", threshold=0.50, window_days=30,
            as_of_date=datetime.date(2025, 2, 1),
        )
        assert result is False

    def test_custom_threshold(self) -> None:
        """Accuracy 0.50 is below a 0.60 threshold -> True."""
        conn = _setup_db()
        # 1 correct, 1 wrong -> accuracy = 0.50
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 6), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 7), close=110.0),
            _make_bar(date=datetime.date(2025, 1, 8), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 9), close=90.0),
        ])
        save_prediction(conn, _make_prediction(
            direction=Direction.UP, date=datetime.date(2025, 1, 6),
        ))
        save_prediction(conn, _make_prediction(
            direction=Direction.UP, date=datetime.date(2025, 1, 8),
        ))

        result = should_retrain(
            conn, "AAPL", threshold=0.60, window_days=30,
            as_of_date=datetime.date(2025, 2, 1),
        )
        assert result is True

    def test_window_days_filters_old_predictions(self) -> None:
        """Predictions outside the window are not counted."""
        conn = _setup_db()
        # Old wrong prediction (outside 30-day window from Feb 1)
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2024, 12, 1), close=100.0),
            _make_bar(date=datetime.date(2024, 12, 2), close=90.0),
        ])
        save_prediction(conn, _make_prediction(
            direction=Direction.UP, date=datetime.date(2024, 12, 1),
        ))

        # Recent correct prediction (inside 30-day window)
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 15), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 16), close=110.0),
        ])
        save_prediction(conn, _make_prediction(
            direction=Direction.UP, date=datetime.date(2025, 1, 15),
        ))

        # Only recent prediction counted -> accuracy 1.0 -> no retrain
        result = should_retrain(
            conn, "AAPL", threshold=0.50, window_days=30,
            as_of_date=datetime.date(2025, 2, 1),
        )
        assert result is False

    def test_ticker_isolation(self) -> None:
        """Accuracy for one ticker does not affect another."""
        conn = _setup_db()
        # AAPL: wrong prediction -> accuracy 0.0
        upsert_bars(conn, [
            _make_bar(ticker="AAPL", date=datetime.date(2025, 1, 6), close=100.0),
            _make_bar(ticker="AAPL", date=datetime.date(2025, 1, 7), close=90.0),
        ])
        save_prediction(conn, _make_prediction(
            ticker="AAPL", direction=Direction.UP,
            date=datetime.date(2025, 1, 6),
        ))

        # MSFT: correct prediction -> accuracy 1.0
        upsert_bars(conn, [
            _make_bar(ticker="MSFT", date=datetime.date(2025, 1, 6), close=200.0),
            _make_bar(ticker="MSFT", date=datetime.date(2025, 1, 7), close=210.0),
        ])
        save_prediction(conn, _make_prediction(
            ticker="MSFT", direction=Direction.UP,
            date=datetime.date(2025, 1, 6),
        ))

        assert should_retrain(
            conn, "AAPL", threshold=0.50, window_days=30,
            as_of_date=datetime.date(2025, 2, 1),
        ) is True
        assert should_retrain(
            conn, "MSFT", threshold=0.50, window_days=30,
            as_of_date=datetime.date(2025, 2, 1),
        ) is False


class TestShouldRetrainLogging:
    """Test that should_retrain emits structured log events."""

    def test_logs_warning_on_trigger(self, caplog: pytest.LogCaptureFixture) -> None:
        """Retrain trigger emits a WARNING-level log with accuracy details."""
        conn = _setup_db()
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 6), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 7), close=90.0),
        ])
        save_prediction(conn, _make_prediction(
            direction=Direction.UP, date=datetime.date(2025, 1, 6),
        ))

        with caplog.at_level(logging.WARNING, logger="smaps.retrainer"):
            should_retrain(
                conn, "AAPL", threshold=0.50, window_days=30,
                as_of_date=datetime.date(2025, 2, 1),
            )

        assert any("retrain_triggered" in msg for msg in caplog.messages)
        assert any("AAPL" in msg for msg in caplog.messages)
        assert any("accuracy=" in msg for msg in caplog.messages)

    def test_logs_info_when_ok(self, caplog: pytest.LogCaptureFixture) -> None:
        """No retrain needed emits an INFO-level log."""
        conn = _setup_db()
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 6), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 7), close=110.0),
        ])
        save_prediction(conn, _make_prediction(
            direction=Direction.UP, date=datetime.date(2025, 1, 6),
        ))

        with caplog.at_level(logging.INFO, logger="smaps.retrainer"):
            should_retrain(
                conn, "AAPL", threshold=0.50, window_days=30,
                as_of_date=datetime.date(2025, 2, 1),
            )

        assert any("result=ok" in msg for msg in caplog.messages)

    def test_logs_info_when_no_predictions(self, caplog: pytest.LogCaptureFixture) -> None:
        """No evaluated predictions emits an INFO-level skip log."""
        conn = _setup_db()

        with caplog.at_level(logging.INFO, logger="smaps.retrainer"):
            should_retrain(
                conn, "AAPL", threshold=0.50, window_days=30,
                as_of_date=datetime.date(2025, 2, 1),
            )

        assert any("no_evaluated_predictions" in msg for msg in caplog.messages)
