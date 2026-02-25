"""Tests for GET /performance endpoint."""

from __future__ import annotations

import datetime
import sqlite3
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from smaps.api import app
from smaps.db import (
    ensure_schema,
    get_connection,
    save_prediction,
    upsert_bars,
)
from smaps.models import Direction, OHLCVBar, PredictionResult


def _make_bar(ticker: str, date: datetime.date, close: float) -> OHLCVBar:
    return OHLCVBar(
        ticker=ticker,
        date=date,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000,
    )


@pytest.fixture()
def perf_db_path(tmp_path):
    """Create a temporary DB with predictions and OHLCV data for evaluation.

    Uses dates within the 90-day window from today so compute_metrics includes them.
    """
    db_path = str(tmp_path / "perf.db")
    conn = get_connection(db_path)
    ensure_schema(conn)

    # Use recent dates (within 90-day window of today)
    pred_date = datetime.date.today() - datetime.timedelta(days=7)
    next_date = pred_date + datetime.timedelta(days=1)

    # AAPL: price goes UP (100 -> 110)
    upsert_bars(
        conn,
        [
            _make_bar("AAPL", pred_date, 100.0),
            _make_bar("AAPL", next_date, 110.0),
        ],
    )
    # Predict UP for AAPL -> correct
    save_prediction(
        conn,
        PredictionResult("AAPL", pred_date, Direction.UP, 0.8, "v1"),
    )

    # MSFT: price goes DOWN (200 -> 190)
    upsert_bars(
        conn,
        [
            _make_bar("MSFT", pred_date, 200.0),
            _make_bar("MSFT", next_date, 190.0),
        ],
    )
    # Predict UP for MSFT -> incorrect (actually DOWN)
    save_prediction(
        conn,
        PredictionResult("MSFT", pred_date, Direction.UP, 0.6, "v1"),
    )

    conn.close()
    return db_path


@pytest.fixture()
def client_with_perf_db(perf_db_path):
    """TestClient with _get_conn patched to use the performance DB file."""

    def _patched():
        conn = sqlite3.connect(perf_db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        ensure_schema(conn)
        return conn

    with patch("smaps.api._get_conn", side_effect=_patched):
        yield TestClient(app)


class TestPerformanceEndpoint:
    """Tests for GET /performance."""

    def test_returns_json_array(self, client_with_perf_db):
        resp = client_with_perf_db.get("/performance")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_one_entry_per_ticker(self, client_with_perf_db):
        resp = client_with_perf_db.get("/performance")
        data = resp.json()
        tickers = [item["ticker"] for item in data]
        assert sorted(tickers) == ["AAPL", "MSFT"]

    def test_response_includes_window_dates(self, client_with_perf_db):
        resp = client_with_perf_db.get("/performance")
        data = resp.json()
        for item in data:
            assert "window_start" in item
            assert "window_end" in item
            # Dates should be ISO format strings
            datetime.date.fromisoformat(item["window_start"])
            datetime.date.fromisoformat(item["window_end"])

    def test_response_includes_accuracy(self, client_with_perf_db):
        resp = client_with_perf_db.get("/performance")
        data = resp.json()
        for item in data:
            assert "accuracy" in item
            assert isinstance(item["accuracy"], float)

    def test_response_includes_precision_recall(self, client_with_perf_db):
        resp = client_with_perf_db.get("/performance")
        data = resp.json()
        required = {"precision_up", "recall_up", "precision_down", "recall_down"}
        for item in data:
            assert required <= set(item.keys())

    def test_aapl_correct_prediction(self, client_with_perf_db):
        """AAPL predicted UP, actual UP -> accuracy 1.0."""
        resp = client_with_perf_db.get("/performance")
        data = resp.json()
        aapl = [item for item in data if item["ticker"] == "AAPL"][0]
        assert aapl["accuracy"] == pytest.approx(1.0)
        assert aapl["evaluated_predictions"] == 1

    def test_msft_incorrect_prediction(self, client_with_perf_db):
        """MSFT predicted UP, actual DOWN -> accuracy 0.0."""
        resp = client_with_perf_db.get("/performance")
        data = resp.json()
        msft = [item for item in data if item["ticker"] == "MSFT"][0]
        assert msft["accuracy"] == pytest.approx(0.0)
        assert msft["evaluated_predictions"] == 1

    def test_total_predictions_count(self, client_with_perf_db):
        resp = client_with_perf_db.get("/performance")
        data = resp.json()
        for item in data:
            assert item["total_predictions"] == 1


class TestPerformanceEmptyDB:
    """Tests with no predictions in DB."""

    def test_empty_db_returns_empty_list(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        conn = get_connection(db_path)
        ensure_schema(conn)
        conn.close()

        def _patched():
            c = sqlite3.connect(db_path, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            ensure_schema(c)
            return c

        with patch("smaps.api._get_conn", side_effect=_patched):
            client = TestClient(app)
            resp = client.get("/performance")

        assert resp.status_code == 200
        assert resp.json() == []


class TestPerformanceUnevaluable:
    """Tests when predictions can't be evaluated (no OHLCV data)."""

    def test_unevaluable_predictions_still_counted(self, tmp_path):
        """Predictions without price data are counted in total but not evaluated."""
        db_path = str(tmp_path / "no_price.db")
        conn = get_connection(db_path)
        ensure_schema(conn)
        # Save prediction with no OHLCV data (use recent date within 90-day window)
        recent_date = datetime.date.today() - datetime.timedelta(days=7)
        save_prediction(
            conn,
            PredictionResult("NOPRICE", recent_date, Direction.UP, 0.5, "v1"),
        )
        conn.close()

        def _patched():
            c = sqlite3.connect(db_path, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            ensure_schema(c)
            return c

        with patch("smaps.api._get_conn", side_effect=_patched):
            client = TestClient(app)
            resp = client.get("/performance")

        data = resp.json()
        assert len(data) == 1
        assert data[0]["ticker"] == "NOPRICE"
        assert data[0]["total_predictions"] == 1
        assert data[0]["evaluated_predictions"] == 0
        assert data[0]["accuracy"] == pytest.approx(0.0)
