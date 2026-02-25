"""Tests for prediction evaluation (US-401)."""

from __future__ import annotations

import datetime

import pytest

from smaps.db import (
    ensure_schema,
    get_connection,
    save_prediction,
    upsert_bars,
)
from smaps.evaluator import evaluate_prediction
from smaps.models import Direction, EvalResult, OHLCVBar, PredictionResult


def _setup_db():
    conn = get_connection(":memory:")
    ensure_schema(conn)
    return conn


def _make_bar(
    ticker: str = "AAPL",
    date: datetime.date | None = None,
    close: float = 100.0,
    open_: float = 99.0,
    high: float = 101.0,
    low: float = 98.0,
    volume: int = 1000,
) -> OHLCVBar:
    return OHLCVBar(
        ticker=ticker,
        date=date or datetime.date(2025, 1, 15),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _make_prediction(
    ticker: str = "AAPL",
    date: datetime.date | None = None,
    direction: Direction = Direction.UP,
    confidence: float = 0.75,
    model_version: str = "v1",
) -> PredictionResult:
    return PredictionResult(
        ticker=ticker,
        prediction_date=date or datetime.date(2025, 1, 15),
        direction=direction,
        confidence=confidence,
        model_version=model_version,
    )


class TestEvaluatePrediction:
    """Core evaluation: compare predicted vs actual direction."""

    def test_correct_up_prediction(self) -> None:
        """Predicted UP, next day close is higher → is_correct=True."""
        conn = _setup_db()
        # Prediction date close=100, next day close=105 → UP
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 15), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 16), close=105.0),
        ])
        pred = _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 15))
        pred_id = save_prediction(conn, pred)

        result = evaluate_prediction(conn, pred_id)

        assert isinstance(result, EvalResult)
        assert result.prediction_id == pred_id
        assert result.actual_direction == Direction.UP
        assert result.is_correct is True

    def test_incorrect_up_prediction(self) -> None:
        """Predicted UP, next day close is lower → is_correct=False."""
        conn = _setup_db()
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 15), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 16), close=95.0),
        ])
        pred = _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 15))
        pred_id = save_prediction(conn, pred)

        result = evaluate_prediction(conn, pred_id)

        assert result.actual_direction == Direction.DOWN
        assert result.is_correct is False

    def test_correct_down_prediction(self) -> None:
        """Predicted DOWN, next day close is lower → is_correct=True."""
        conn = _setup_db()
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 15), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 16), close=90.0),
        ])
        pred = _make_prediction(direction=Direction.DOWN, date=datetime.date(2025, 1, 15))
        pred_id = save_prediction(conn, pred)

        result = evaluate_prediction(conn, pred_id)

        assert result.actual_direction == Direction.DOWN
        assert result.is_correct is True

    def test_incorrect_down_prediction(self) -> None:
        """Predicted DOWN, next day close is higher → is_correct=False."""
        conn = _setup_db()
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 15), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 16), close=110.0),
        ])
        pred = _make_prediction(direction=Direction.DOWN, date=datetime.date(2025, 1, 15))
        pred_id = save_prediction(conn, pred)

        result = evaluate_prediction(conn, pred_id)

        assert result.actual_direction == Direction.UP
        assert result.is_correct is False

    def test_flat_day_counts_as_up(self) -> None:
        """When next day close == prediction day close, actual = UP."""
        conn = _setup_db()
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 15), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 16), close=100.0),
        ])
        pred = _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 15))
        pred_id = save_prediction(conn, pred)

        result = evaluate_prediction(conn, pred_id)

        assert result.actual_direction == Direction.UP
        assert result.is_correct is True

    def test_evaluated_at_is_datetime_with_tz(self) -> None:
        """evaluated_at should be a timezone-aware UTC datetime."""
        conn = _setup_db()
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 15), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 16), close=105.0),
        ])
        pred = _make_prediction(date=datetime.date(2025, 1, 15))
        pred_id = save_prediction(conn, pred)

        result = evaluate_prediction(conn, pred_id)

        assert isinstance(result.evaluated_at, datetime.datetime)
        assert result.evaluated_at.tzinfo is not None


class TestWeekendHolidayHandling:
    """Verify that weekends/holidays (non-trading days) are skipped."""

    def test_skips_weekend(self) -> None:
        """Friday prediction → Monday outcome (no Sat/Sun bars)."""
        conn = _setup_db()
        # Friday 2025-01-17, next trading day is Monday 2025-01-20
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 17), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 20), close=108.0),
        ])
        pred = _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 17))
        pred_id = save_prediction(conn, pred)

        result = evaluate_prediction(conn, pred_id)

        assert result.actual_direction == Direction.UP
        assert result.is_correct is True

    def test_skips_holiday_gap(self) -> None:
        """Multi-day gap (e.g. 3-day weekend) is handled."""
        conn = _setup_db()
        # Thursday 2025-01-16, skip Fri+Sat+Sun, next bar is Tuesday 2025-01-21
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 16), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 21), close=90.0),
        ])
        pred = _make_prediction(direction=Direction.DOWN, date=datetime.date(2025, 1, 16))
        pred_id = save_prediction(conn, pred)

        result = evaluate_prediction(conn, pred_id)

        assert result.actual_direction == Direction.DOWN
        assert result.is_correct is True


class TestErrorCases:
    """Verify error handling for missing data."""

    def test_nonexistent_prediction_raises(self) -> None:
        conn = _setup_db()
        with pytest.raises(ValueError, match="not found"):
            evaluate_prediction(conn, 9999)

    def test_no_price_on_prediction_date_raises(self) -> None:
        """Prediction date has no OHLCV bar → ValueError."""
        conn = _setup_db()
        # Only insert the next-day bar, not the prediction date bar
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 16), close=105.0),
        ])
        pred = _make_prediction(date=datetime.date(2025, 1, 15))
        pred_id = save_prediction(conn, pred)

        with pytest.raises(ValueError, match="No price data on prediction date"):
            evaluate_prediction(conn, pred_id)

    def test_no_next_day_price_raises(self) -> None:
        """No trading day found after prediction date → ValueError."""
        conn = _setup_db()
        # Only the prediction date bar, no next day
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 15), close=100.0),
        ])
        pred = _make_prediction(date=datetime.date(2025, 1, 15))
        pred_id = save_prediction(conn, pred)

        with pytest.raises(ValueError, match="No next-day price data"):
            evaluate_prediction(conn, pred_id)
