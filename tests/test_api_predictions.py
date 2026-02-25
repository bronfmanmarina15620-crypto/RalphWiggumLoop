"""Tests for GET /predictions/latest endpoint."""

from __future__ import annotations

import datetime
import sqlite3
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from smaps.api import app
from smaps.db import ensure_schema, get_connection, save_prediction
from smaps.models import Direction, PredictionResult


@pytest.fixture()
def populated_db_path(tmp_path):
    """Create a temporary DB file with predictions and return its path."""
    db_path = str(tmp_path / "test.db")
    conn = get_connection(db_path)
    ensure_schema(conn)

    save_prediction(
        conn,
        PredictionResult(
            ticker="AAPL",
            prediction_date=datetime.date(2025, 1, 10),
            direction=Direction.UP,
            confidence=0.75,
            model_version="v1",
        ),
    )
    save_prediction(
        conn,
        PredictionResult(
            ticker="AAPL",
            prediction_date=datetime.date(2025, 1, 11),
            direction=Direction.DOWN,
            confidence=0.62,
            model_version="v2",
        ),
    )
    save_prediction(
        conn,
        PredictionResult(
            ticker="MSFT",
            prediction_date=datetime.date(2025, 1, 11),
            direction=Direction.UP,
            confidence=0.88,
            model_version="v1",
        ),
    )
    conn.close()
    return db_path


@pytest.fixture()
def client_with_db(populated_db_path):
    """TestClient with _get_conn patched to use the populated DB file."""
    def _patched():
        conn = sqlite3.connect(populated_db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        ensure_schema(conn)
        return conn

    with patch("smaps.api._get_conn", side_effect=_patched):
        yield TestClient(app)


@pytest.fixture()
def client():
    return TestClient(app)


class TestPredictionsLatest:
    """Tests for GET /predictions/latest."""

    def test_returns_json_array(self, client_with_db):
        resp = client_with_db.get("/predictions/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_returns_latest_per_ticker(self, client_with_db):
        resp = client_with_db.get("/predictions/latest")
        data = resp.json()
        tickers = [p["ticker"] for p in data]
        assert sorted(tickers) == ["AAPL", "MSFT"]

    def test_latest_aapl_is_most_recent(self, client_with_db):
        resp = client_with_db.get("/predictions/latest")
        data = resp.json()
        aapl = [p for p in data if p["ticker"] == "AAPL"][0]
        assert aapl["prediction_date"] == "2025-01-11"
        assert aapl["direction"] == "DOWN"
        assert aapl["model_version"] == "v2"

    def test_filter_by_ticker(self, client_with_db):
        resp = client_with_db.get("/predictions/latest", params={"ticker": "MSFT"})
        data = resp.json()
        assert len(data) == 1
        assert data[0]["ticker"] == "MSFT"
        assert data[0]["direction"] == "UP"
        assert data[0]["confidence"] == pytest.approx(0.88)

    def test_filter_unknown_ticker_returns_empty(self, client_with_db):
        resp = client_with_db.get("/predictions/latest", params={"ticker": "UNKNOWN"})
        data = resp.json()
        assert data == []

    def test_response_includes_required_fields(self, client_with_db):
        resp = client_with_db.get("/predictions/latest")
        data = resp.json()
        required = {"ticker", "prediction_date", "direction", "confidence", "model_version"}
        for item in data:
            assert required <= set(item.keys())

    def test_direction_is_up_or_down(self, client_with_db):
        resp = client_with_db.get("/predictions/latest")
        for item in resp.json():
            assert item["direction"] in ("UP", "DOWN")

    def test_confidence_in_range(self, client_with_db):
        resp = client_with_db.get("/predictions/latest")
        for item in resp.json():
            assert 0.0 <= item["confidence"] <= 1.0


class TestPredictionsLatestEmptyDB:
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
            resp = client.get("/predictions/latest")

        assert resp.status_code == 200
        assert resp.json() == []


class TestGetLatestPredictionsDB:
    """Unit tests for get_latest_predictions DB function."""

    def test_no_predictions(self):
        conn = get_connection()
        ensure_schema(conn)
        from smaps.db import get_latest_predictions

        result = get_latest_predictions(conn)
        assert result == []
        conn.close()

    def test_latest_per_ticker(self):
        conn = get_connection()
        ensure_schema(conn)
        from smaps.db import get_latest_predictions

        save_prediction(
            conn,
            PredictionResult("T1", datetime.date(2025, 1, 1), Direction.UP, 0.5, "v1"),
        )
        save_prediction(
            conn,
            PredictionResult("T1", datetime.date(2025, 1, 2), Direction.DOWN, 0.6, "v2"),
        )
        save_prediction(
            conn,
            PredictionResult("T2", datetime.date(2025, 1, 1), Direction.UP, 0.7, "v1"),
        )

        result = get_latest_predictions(conn)
        assert len(result) == 2
        tickers = {r.ticker for r in result}
        assert tickers == {"T1", "T2"}

        t1 = [r for r in result if r.ticker == "T1"][0]
        assert t1.direction == Direction.DOWN
        assert t1.model_version == "v2"

        conn.close()

    def test_filter_by_ticker(self):
        conn = get_connection()
        ensure_schema(conn)
        from smaps.db import get_latest_predictions

        save_prediction(
            conn,
            PredictionResult("AA", datetime.date(2025, 1, 1), Direction.UP, 0.5, "v1"),
        )
        save_prediction(
            conn,
            PredictionResult("BB", datetime.date(2025, 1, 1), Direction.DOWN, 0.6, "v1"),
        )

        result = get_latest_predictions(conn, ticker="AA")
        assert len(result) == 1
        assert result[0].ticker == "AA"

        conn.close()
