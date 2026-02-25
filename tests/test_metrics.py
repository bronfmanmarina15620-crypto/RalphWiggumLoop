"""Tests for rolling accuracy metrics (US-402)."""

from __future__ import annotations

import datetime
import json

import pytest

from smaps.db import (
    ensure_schema,
    get_connection,
    save_prediction,
    upsert_bars,
)
from smaps.evaluator import compute_metrics
from smaps.models import Direction, MetricsReport, OHLCVBar, PredictionResult


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


class TestComputeMetrics:
    """Test rolling accuracy metric computation."""

    def test_returns_metrics_report(self) -> None:
        """compute_metrics should return MetricsReport."""
        conn = _setup_db()
        result = compute_metrics(
            conn, "AAPL", window_days=90, as_of_date=datetime.date(2025, 3, 1)
        )
        assert isinstance(result, MetricsReport)

    def test_perfect_accuracy(self) -> None:
        """All correct predictions -> accuracy = 1.0."""
        conn = _setup_db()
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 6), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 7), close=110.0),
            _make_bar(date=datetime.date(2025, 1, 8), close=105.0),
            _make_bar(date=datetime.date(2025, 1, 9), close=115.0),
            _make_bar(date=datetime.date(2025, 1, 10), close=108.0),
            _make_bar(date=datetime.date(2025, 1, 13), close=120.0),
        ])
        # Pred 1: Jan 6 -> Jan 7: 100->110 UP
        save_prediction(conn, _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 6)))
        # Pred 2: Jan 8 -> Jan 9: 105->115 UP
        save_prediction(conn, _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 8)))
        # Pred 3: Jan 10 -> Jan 13 (skip weekend): 108->120 UP
        save_prediction(conn, _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 10)))

        result = compute_metrics(
            conn, "AAPL", window_days=90, as_of_date=datetime.date(2025, 2, 1)
        )
        assert result.accuracy == 1.0
        assert result.evaluated_predictions == 3

    def test_zero_accuracy(self) -> None:
        """All wrong predictions -> accuracy = 0.0."""
        conn = _setup_db()
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 6), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 7), close=90.0),
            _make_bar(date=datetime.date(2025, 1, 8), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 9), close=85.0),
        ])
        save_prediction(conn, _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 6)))
        save_prediction(conn, _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 8)))

        result = compute_metrics(
            conn, "AAPL", window_days=90, as_of_date=datetime.date(2025, 2, 1)
        )
        assert result.accuracy == 0.0

    def test_mixed_accuracy(self) -> None:
        """Known correct and incorrect predictions -> verify exact metrics."""
        conn = _setup_db()

        # Prediction 1: predict UP, actual UP (TP_up)
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 6), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 7), close=110.0),
        ])
        save_prediction(conn, _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 6)))

        # Prediction 2: predict UP, actual DOWN (FP_up)
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 8), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 9), close=90.0),
        ])
        save_prediction(conn, _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 8)))

        # Prediction 3: predict DOWN, actual DOWN (TP_down)
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 10), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 13), close=90.0),
        ])
        save_prediction(conn, _make_prediction(direction=Direction.DOWN, date=datetime.date(2025, 1, 10)))

        # Prediction 4: predict DOWN, actual UP (FN_up)
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 14), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 15), close=110.0),
        ])
        save_prediction(conn, _make_prediction(direction=Direction.DOWN, date=datetime.date(2025, 1, 14)))

        result = compute_metrics(
            conn, "AAPL", window_days=90, as_of_date=datetime.date(2025, 2, 1)
        )

        # TP_up=1, FP_up=1, FN_up=1, TP_down=1
        assert result.accuracy == pytest.approx(0.5)
        assert result.precision_up == pytest.approx(0.5)
        assert result.recall_up == pytest.approx(0.5)
        assert result.precision_down == pytest.approx(0.5)
        assert result.recall_down == pytest.approx(0.5)
        assert result.total_predictions == 4
        assert result.evaluated_predictions == 4

    def test_precision_recall_asymmetry(self) -> None:
        """Asymmetric predictions verify precision != recall."""
        conn = _setup_db()

        # 2 correct UP, 1 incorrect UP → TP_up=2, FP_up=1, FN_up=0, TP_down=0
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 6), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 7), close=110.0),
            _make_bar(date=datetime.date(2025, 1, 8), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 9), close=105.0),
            _make_bar(date=datetime.date(2025, 1, 10), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 13), close=90.0),
        ])
        save_prediction(conn, _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 6)))
        save_prediction(conn, _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 8)))
        save_prediction(conn, _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 10)))

        result = compute_metrics(
            conn, "AAPL", window_days=90, as_of_date=datetime.date(2025, 2, 1)
        )

        assert result.precision_up == pytest.approx(2 / 3)
        assert result.recall_up == pytest.approx(1.0)  # all actual UPs predicted correctly
        assert result.precision_down == pytest.approx(0.0)  # no DOWN predictions made
        assert result.recall_down == pytest.approx(0.0)  # actual DOWN not caught

    def test_no_predictions_returns_zeros(self) -> None:
        """No predictions in window -> all metrics zero."""
        conn = _setup_db()
        result = compute_metrics(
            conn, "AAPL", window_days=90, as_of_date=datetime.date(2025, 2, 1)
        )

        assert result.accuracy == 0.0
        assert result.precision_up == 0.0
        assert result.recall_up == 0.0
        assert result.precision_down == 0.0
        assert result.recall_down == 0.0
        assert result.total_predictions == 0
        assert result.evaluated_predictions == 0

    def test_window_filters_old_predictions(self) -> None:
        """Predictions outside the window are not counted."""
        conn = _setup_db()

        # Old prediction (outside 30-day window from Jan 31)
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2024, 12, 1), close=100.0),
            _make_bar(date=datetime.date(2024, 12, 2), close=110.0),
        ])
        save_prediction(conn, _make_prediction(direction=Direction.UP, date=datetime.date(2024, 12, 1)))

        # Recent prediction (inside 30-day window from Jan 31)
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 15), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 16), close=110.0),
        ])
        save_prediction(conn, _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 15)))

        result = compute_metrics(
            conn, "AAPL", window_days=30, as_of_date=datetime.date(2025, 1, 31)
        )

        assert result.total_predictions == 1
        assert result.evaluated_predictions == 1

    def test_ticker_isolation(self) -> None:
        """Predictions for other tickers are not included."""
        conn = _setup_db()

        upsert_bars(conn, [
            _make_bar(ticker="AAPL", date=datetime.date(2025, 1, 15), close=100.0),
            _make_bar(ticker="AAPL", date=datetime.date(2025, 1, 16), close=110.0),
        ])
        save_prediction(conn, _make_prediction(ticker="AAPL", direction=Direction.UP, date=datetime.date(2025, 1, 15)))

        upsert_bars(conn, [
            _make_bar(ticker="MSFT", date=datetime.date(2025, 1, 15), close=200.0),
            _make_bar(ticker="MSFT", date=datetime.date(2025, 1, 16), close=190.0),
        ])
        save_prediction(conn, _make_prediction(ticker="MSFT", direction=Direction.DOWN, date=datetime.date(2025, 1, 15)))

        result = compute_metrics(
            conn, "AAPL", window_days=90, as_of_date=datetime.date(2025, 2, 1)
        )

        assert result.total_predictions == 1
        assert result.ticker == "AAPL"

    def test_skips_unevaluable_predictions(self) -> None:
        """Predictions without price data are skipped gracefully."""
        conn = _setup_db()

        # Evaluable prediction (has price data)
        upsert_bars(conn, [
            _make_bar(date=datetime.date(2025, 1, 15), close=100.0),
            _make_bar(date=datetime.date(2025, 1, 16), close=110.0),
        ])
        save_prediction(conn, _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 15)))

        # Unevaluable prediction (no price data)
        save_prediction(conn, _make_prediction(direction=Direction.UP, date=datetime.date(2025, 1, 20)))

        result = compute_metrics(
            conn, "AAPL", window_days=90, as_of_date=datetime.date(2025, 2, 1)
        )

        assert result.total_predictions == 2
        assert result.evaluated_predictions == 1

    def test_window_start_and_end(self) -> None:
        """MetricsReport includes correct window boundaries."""
        conn = _setup_db()
        result = compute_metrics(
            conn, "AAPL", window_days=90, as_of_date=datetime.date(2025, 3, 31)
        )

        assert result.window_start == datetime.date(2024, 12, 31)
        assert result.window_end == datetime.date(2025, 3, 31)


class TestMetricsReportSerialization:
    """Test MetricsReport JSON serialization."""

    def test_to_dict_returns_dict(self) -> None:
        report = MetricsReport(
            ticker="AAPL",
            window_start=datetime.date(2025, 1, 1),
            window_end=datetime.date(2025, 3, 31),
            accuracy=0.65,
            precision_up=0.7,
            recall_up=0.6,
            precision_down=0.6,
            recall_down=0.7,
            total_predictions=100,
            evaluated_predictions=95,
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["ticker"] == "AAPL"
        assert d["accuracy"] == 0.65

    def test_to_dict_json_serializable(self) -> None:
        """to_dict output should be directly JSON-serializable."""
        report = MetricsReport(
            ticker="AAPL",
            window_start=datetime.date(2025, 1, 1),
            window_end=datetime.date(2025, 3, 31),
            accuracy=0.65,
            precision_up=0.7,
            recall_up=0.6,
            precision_down=0.6,
            recall_down=0.7,
            total_predictions=100,
            evaluated_predictions=95,
        )
        json_str = json.dumps(report.to_dict())
        parsed = json.loads(json_str)
        assert parsed["ticker"] == "AAPL"
        assert parsed["window_start"] == "2025-01-01"
        assert parsed["accuracy"] == 0.65

    def test_to_dict_dates_as_iso_strings(self) -> None:
        report = MetricsReport(
            ticker="AAPL",
            window_start=datetime.date(2025, 1, 1),
            window_end=datetime.date(2025, 3, 31),
            accuracy=0.5,
            precision_up=0.5,
            recall_up=0.5,
            precision_down=0.5,
            recall_down=0.5,
            total_predictions=10,
            evaluated_predictions=10,
        )
        d = report.to_dict()
        assert d["window_start"] == "2025-01-01"
        assert d["window_end"] == "2025-03-31"
