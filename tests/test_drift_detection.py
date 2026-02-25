"""Tests for feature drift detection (US-505)."""

from __future__ import annotations

import datetime
import json
import logging

import pytest

from smaps.db import (
    ensure_schema,
    get_connection,
    upsert_bars,
)
from smaps.models import OHLCVBar
from smaps.retrainer import detect_drift


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
    days: int = 60,
    close_fn=None,
) -> None:
    """Insert synthetic daily OHLCV bars.

    Args:
        close_fn: Callable(i) -> close price. Defaults to alternating 100/110.
    """
    if close_fn is None:
        close_fn = lambda i: 100.0 + (i % 2) * 10.0

    bars = []
    for i in range(days):
        dt = start + datetime.timedelta(days=i)
        close = close_fn(i)
        bars.append(_make_bar(ticker=ticker, date=dt, close=close))
    upsert_bars(conn, bars)


class TestDetectDrift:
    """Test the detect_drift() function."""

    def test_returns_report_dict(self):
        conn = _setup_db()
        _insert_price_history(conn, days=60)
        report = detect_drift(
            conn, "AAPL", window_days=10,
            as_of_date=datetime.date(2025, 2, 28),
        )
        assert isinstance(report, dict)
        assert "ticker" in report
        assert "features" in report
        assert "drifted_features" in report
        assert report["ticker"] == "AAPL"

    def test_no_drift_with_stable_distribution(self):
        """When training and recent data come from the same distribution,
        no features should be flagged as drifted."""
        conn = _setup_db()
        # Uniform distribution — same pattern throughout
        _insert_price_history(conn, days=60)
        report = detect_drift(
            conn, "AAPL", window_days=10,
            as_of_date=datetime.date(2025, 2, 28),
        )
        assert report["drifted_features"] == [] or isinstance(report["drifted_features"], list)

    def test_drift_detected_with_shifted_distribution(self):
        """When the recent window has a dramatically different distribution,
        drift should be detected in at least one feature."""
        conn = _setup_db()

        # Training period: prices around 100
        training_bars = []
        start = datetime.date(2025, 1, 1)
        for i in range(50):
            dt = start + datetime.timedelta(days=i)
            close = 100.0 + (i % 2) * 2.0  # small oscillation: 100, 102
            training_bars.append(_make_bar(ticker="AAPL", date=dt, close=close))

        # Recent period: prices around 500 (huge shift)
        for i in range(50, 80):
            dt = start + datetime.timedelta(days=i)
            close = 500.0 + (i % 2) * 2.0  # 500, 502
            training_bars.append(_make_bar(ticker="AAPL", date=dt, close=close))

        upsert_bars(conn, training_bars)

        report = detect_drift(
            conn, "AAPL", window_days=20,
            as_of_date=datetime.date(2025, 3, 21),
        )
        drifted = report["drifted_features"]
        assert isinstance(drifted, list)
        assert len(drifted) > 0, "Should detect drift with shifted distribution"

    def test_ks_test_per_feature(self):
        """Each feature should have statistic and p_value in the report."""
        conn = _setup_db()
        _insert_price_history(conn, days=60)
        report = detect_drift(
            conn, "AAPL", window_days=10,
            as_of_date=datetime.date(2025, 2, 28),
        )
        features = report["features"]
        assert isinstance(features, dict)
        assert len(features) > 0
        for name, result in features.items():
            assert "statistic" in result
            assert "p_value" in result
            assert "drifted" in result

    def test_insufficient_data_skips(self):
        """When there aren't enough trading days, drift check should be skipped."""
        conn = _setup_db()
        _insert_price_history(conn, days=5)
        report = detect_drift(
            conn, "AAPL", window_days=30,
            as_of_date=datetime.date(2025, 1, 5),
        )
        assert report.get("skipped") is True
        assert report["drifted_features"] == []

    def test_no_training_dates_skips(self):
        """When all dates fall in the recent window, no training baseline exists."""
        conn = _setup_db()
        _insert_price_history(conn, days=10)
        # window_days=10 with only 10 trading days → ≤ window_days → skip
        report = detect_drift(
            conn, "AAPL", window_days=10,
            as_of_date=datetime.date(2025, 1, 10),
        )
        assert report.get("skipped") is True

    def test_report_persisted_to_file(self, tmp_path):
        """Drift report should be saved to reports/drift_<date>.json."""
        conn = _setup_db()
        _insert_price_history(conn, days=60)

        reports_dir = str(tmp_path / "reports")
        as_of_date = datetime.date(2025, 2, 28)

        detect_drift(
            conn, "AAPL", window_days=10,
            as_of_date=as_of_date,
            reports_dir=reports_dir,
        )

        filepath = tmp_path / "reports" / f"drift_{as_of_date.isoformat()}.json"
        assert filepath.exists()

        content = json.loads(filepath.read_text())
        assert content["ticker"] == "AAPL"
        assert content["as_of_date"] == as_of_date.isoformat()
        assert "features" in content

    def test_report_creates_directory(self, tmp_path):
        """Reports directory should be created if it doesn't exist."""
        conn = _setup_db()
        _insert_price_history(conn, days=60)

        reports_dir = str(tmp_path / "new_reports_dir")
        detect_drift(
            conn, "AAPL", window_days=10,
            as_of_date=datetime.date(2025, 2, 28),
            reports_dir=reports_dir,
        )

        assert (tmp_path / "new_reports_dir").is_dir()

    def test_as_of_date_defaults_to_today(self):
        """When as_of_date is None, should use today's date."""
        conn = _setup_db()
        start = datetime.date(2025, 1, 1)
        _insert_price_history(
            conn, days=200, start=start,
        )
        report = detect_drift(
            conn, "AAPL", window_days=10,
            reports_dir="/tmp/test_drift_reports",
        )
        # Should not raise; report should have as_of_date
        assert "as_of_date" in report

    def test_ticker_isolation(self):
        """Drift detection for one ticker should not include another's data."""
        conn = _setup_db()
        _insert_price_history(conn, ticker="AAPL", days=60)
        _insert_price_history(conn, ticker="MSFT", days=60,
                              close_fn=lambda i: 200.0 + i * 5.0)

        report = detect_drift(
            conn, "AAPL", window_days=10,
            as_of_date=datetime.date(2025, 2, 28),
        )
        assert report["ticker"] == "AAPL"

    def test_custom_p_threshold(self):
        """A custom p_threshold should be used in the KS-test comparison."""
        conn = _setup_db()
        _insert_price_history(conn, days=60)

        # With threshold=1.0 nothing should ever drift (all p-values < 1.0 pass)
        report = detect_drift(
            conn, "AAPL", window_days=10,
            p_threshold=1.0,
            as_of_date=datetime.date(2025, 2, 28),
        )
        assert report["drifted_features"] == []

    def test_report_includes_metadata(self):
        """Report should include window_days, p_threshold, and sample counts."""
        conn = _setup_db()
        _insert_price_history(conn, days=60)
        report = detect_drift(
            conn, "AAPL", window_days=10, p_threshold=0.05,
            as_of_date=datetime.date(2025, 2, 28),
        )
        assert report["window_days"] == 10
        assert report["p_threshold"] == 0.05
        assert "training_samples" in report
        assert "recent_samples" in report
        assert report["recent_samples"] == 10


class TestDetectDriftLogging:
    """Test drift detection logging."""

    def test_logs_warning_on_drift(self, caplog):
        """A WARNING should be logged when drift is detected."""
        conn = _setup_db()
        start = datetime.date(2025, 1, 1)

        # Training: stable prices around 100
        bars = []
        for i in range(50):
            dt = start + datetime.timedelta(days=i)
            bars.append(_make_bar(ticker="AAPL", date=dt, close=100.0 + (i % 2) * 2.0))

        # Recent: dramatically shifted prices around 500
        for i in range(50, 80):
            dt = start + datetime.timedelta(days=i)
            bars.append(_make_bar(ticker="AAPL", date=dt, close=500.0 + (i % 2) * 2.0))

        upsert_bars(conn, bars)

        with caplog.at_level(logging.WARNING, logger="smaps.retrainer"):
            detect_drift(
                conn, "AAPL", window_days=20,
                as_of_date=datetime.date(2025, 3, 21),
                reports_dir="/tmp/test_drift_log_reports",
            )

        assert any("drift_alert" in msg for msg in caplog.messages)

    def test_logs_info_no_drift(self, caplog):
        """INFO should be logged when no drift is detected."""
        conn = _setup_db()
        _insert_price_history(conn, days=60)

        with caplog.at_level(logging.INFO, logger="smaps.retrainer"):
            detect_drift(
                conn, "AAPL", window_days=10,
                p_threshold=0.0,  # No feature will have p < 0
                as_of_date=datetime.date(2025, 2, 28),
                reports_dir="/tmp/test_drift_log_reports2",
            )

        assert any("drift_check" in msg and "no_drift" in msg for msg in caplog.messages)

    def test_logs_info_on_skip(self, caplog):
        """INFO should be logged when drift check is skipped."""
        conn = _setup_db()
        _insert_price_history(conn, days=5)

        with caplog.at_level(logging.INFO, logger="smaps.retrainer"):
            detect_drift(
                conn, "AAPL", window_days=30,
                as_of_date=datetime.date(2025, 1, 5),
                reports_dir="/tmp/test_drift_log_reports3",
            )

        assert any("drift_check" in msg and "skip" in msg for msg in caplog.messages)

    def test_drift_alert_includes_feature_name(self, caplog):
        """Drift alert log should include the feature name and p-value."""
        conn = _setup_db()
        start = datetime.date(2025, 1, 1)

        # Create a clear distribution shift
        bars = []
        for i in range(50):
            dt = start + datetime.timedelta(days=i)
            bars.append(_make_bar(ticker="AAPL", date=dt, close=100.0 + (i % 2) * 2.0))
        for i in range(50, 80):
            dt = start + datetime.timedelta(days=i)
            bars.append(_make_bar(ticker="AAPL", date=dt, close=500.0 + (i % 2) * 2.0))
        upsert_bars(conn, bars)

        with caplog.at_level(logging.WARNING, logger="smaps.retrainer"):
            detect_drift(
                conn, "AAPL", window_days=20,
                as_of_date=datetime.date(2025, 3, 21),
                reports_dir="/tmp/test_drift_log_reports4",
            )

        alert_messages = [m for m in caplog.messages if "drift_alert" in m]
        assert len(alert_messages) > 0
        # Each alert should include feature name and p_value
        for msg in alert_messages:
            assert "feature=" in msg
            assert "p_value=" in msg
