"""Tests for GET /dashboard and GET /retrain-info endpoints."""

from __future__ import annotations

import datetime
import json
import sqlite3
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from smaps.api import app
from smaps.db import ensure_schema, get_connection, get_last_retrain_dates, save_prediction
from smaps.models import Direction, PredictionResult


@pytest.fixture()
def empty_db_path(tmp_path):
    """Create an empty DB and return its path."""
    db_path = str(tmp_path / "empty.db")
    conn = get_connection(db_path)
    ensure_schema(conn)
    conn.close()
    return db_path


@pytest.fixture()
def populated_db_path(tmp_path):
    """Create a DB with predictions and a model_registry entry, return its path."""
    db_path = str(tmp_path / "test.db")
    conn = get_connection(db_path)
    ensure_schema(conn)

    # Add predictions
    save_prediction(
        conn,
        PredictionResult(
            ticker="AAPL",
            prediction_date=datetime.date.today() - datetime.timedelta(days=3),
            direction=Direction.UP,
            confidence=0.75,
            model_version="v1",
        ),
    )
    save_prediction(
        conn,
        PredictionResult(
            ticker="MSFT",
            prediction_date=datetime.date.today() - datetime.timedelta(days=2),
            direction=Direction.DOWN,
            confidence=0.62,
            model_version="v2",
        ),
    )

    # Add model_registry entries for retrain info
    trained_at = datetime.datetime(2025, 6, 15, 10, 30, 0, tzinfo=datetime.timezone.utc)
    conn.execute(
        "INSERT INTO model_registry (ticker, version, trained_at, metrics_json, artifact_path) "
        "VALUES (?, ?, ?, ?, ?)",
        ("AAPL", 1, trained_at.isoformat(), json.dumps({"accuracy": 0.65}), "models/AAPL_v1.joblib"),
    )
    trained_at_2 = datetime.datetime(2025, 7, 1, 8, 0, 0, tzinfo=datetime.timezone.utc)
    conn.execute(
        "INSERT INTO model_registry (ticker, version, trained_at, metrics_json, artifact_path) "
        "VALUES (?, ?, ?, ?, ?)",
        ("AAPL", 2, trained_at_2.isoformat(), json.dumps({"accuracy": 0.70}), "models/AAPL_v2.joblib"),
    )
    conn.execute(
        "INSERT INTO model_registry (ticker, version, trained_at, metrics_json, artifact_path) "
        "VALUES (?, ?, ?, ?, ?)",
        ("MSFT", 1, trained_at.isoformat(), json.dumps({"accuracy": 0.55}), "models/MSFT_v1.joblib"),
    )
    conn.commit()
    conn.close()
    return db_path


def _make_client(db_path: str):
    """Create a TestClient with _get_conn patched to use the given DB file."""
    def _patched():
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        ensure_schema(conn)
        return conn

    return patch("smaps.api._get_conn", side_effect=_patched)


class TestDashboardEndpoint:
    """Tests for GET /dashboard."""

    def test_returns_html(self, empty_db_path):
        with _make_client(empty_db_path):
            client = TestClient(app)
            resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_contains_title(self, empty_db_path):
        with _make_client(empty_db_path):
            client = TestClient(app)
            resp = client.get("/dashboard")
        assert "SMAPS Dashboard" in resp.text

    def test_contains_predictions_section(self, empty_db_path):
        with _make_client(empty_db_path):
            client = TestClient(app)
            resp = client.get("/dashboard")
        assert 'id="predictions-table"' in resp.text

    def test_contains_accuracy_section(self, empty_db_path):
        with _make_client(empty_db_path):
            client = TestClient(app)
            resp = client.get("/dashboard")
        assert 'id="accuracy-chart"' in resp.text

    def test_contains_retrain_section(self, empty_db_path):
        with _make_client(empty_db_path):
            client = TestClient(app)
            resp = client.get("/dashboard")
        assert 'id="retrain-info"' in resp.text

    def test_contains_fetch_calls(self, empty_db_path):
        with _make_client(empty_db_path):
            client = TestClient(app)
            resp = client.get("/dashboard")
        # Verify auto-refresh: JS fetches data on load
        assert "/predictions/latest" in resp.text
        assert "/performance" in resp.text
        assert "/retrain-info" in resp.text


class TestRetrainInfoEndpoint:
    """Tests for GET /retrain-info."""

    def test_returns_dict(self, populated_db_path):
        with _make_client(populated_db_path):
            client = TestClient(app)
            resp = client.get("/retrain-info")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_returns_latest_per_ticker(self, populated_db_path):
        with _make_client(populated_db_path):
            client = TestClient(app)
            resp = client.get("/retrain-info")
        data = resp.json()
        assert "AAPL" in data
        assert "MSFT" in data
        # AAPL should have the v2 trained_at (2025-07-01), not v1
        assert "2025-07-01" in data["AAPL"]

    def test_empty_db_returns_empty_dict(self, empty_db_path):
        with _make_client(empty_db_path):
            client = TestClient(app)
            resp = client.get("/retrain-info")
        assert resp.status_code == 200
        assert resp.json() == {}


class TestGetLastRetrainDatesDB:
    """Unit tests for get_last_retrain_dates DB function."""

    def test_no_models_returns_empty_dict(self):
        conn = get_connection()
        ensure_schema(conn)
        result = get_last_retrain_dates(conn)
        assert result == {}
        conn.close()

    def test_returns_latest_per_ticker(self):
        conn = get_connection()
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO model_registry (ticker, version, trained_at, metrics_json, artifact_path) "
            "VALUES (?, ?, ?, ?, ?)",
            ("XX", 1, "2025-01-01T00:00:00+00:00", "{}", "models/XX_v1.joblib"),
        )
        conn.execute(
            "INSERT INTO model_registry (ticker, version, trained_at, metrics_json, artifact_path) "
            "VALUES (?, ?, ?, ?, ?)",
            ("XX", 2, "2025-02-01T00:00:00+00:00", "{}", "models/XX_v2.joblib"),
        )
        conn.execute(
            "INSERT INTO model_registry (ticker, version, trained_at, metrics_json, artifact_path) "
            "VALUES (?, ?, ?, ?, ?)",
            ("YY", 1, "2025-03-01T00:00:00+00:00", "{}", "models/YY_v1.joblib"),
        )
        conn.commit()

        result = get_last_retrain_dates(conn)
        assert len(result) == 2
        assert "2025-02-01" in result["XX"]
        assert "2025-03-01" in result["YY"]
        conn.close()

    def test_sorted_by_ticker(self):
        conn = get_connection()
        ensure_schema(conn)
        for ticker in ["CC", "AA", "BB"]:
            conn.execute(
                "INSERT INTO model_registry (ticker, version, trained_at, metrics_json, artifact_path) "
                "VALUES (?, ?, ?, ?, ?)",
                (ticker, 1, "2025-01-01T00:00:00+00:00", "{}", f"models/{ticker}_v1.joblib"),
            )
        conn.commit()

        result = get_last_retrain_dates(conn)
        assert list(result.keys()) == ["AA", "BB", "CC"]
        conn.close()
