"""Tests for prediction persistence (US-305)."""

from __future__ import annotations

import datetime

from smaps.db import (
    PredictionRecord,
    ensure_schema,
    get_connection,
    load_prediction,
    save_prediction,
)
from smaps.models import Direction, PredictionResult


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


def _setup_db():
    conn = get_connection(":memory:")
    ensure_schema(conn)
    return conn


class TestSavePrediction:
    def test_save_returns_autoincrement_id(self) -> None:
        conn = _setup_db()
        pred = _make_prediction()
        row_id = save_prediction(conn, pred)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_save_multiple_returns_sequential_ids(self) -> None:
        conn = _setup_db()
        id1 = save_prediction(conn, _make_prediction(ticker="AAPL"))
        id2 = save_prediction(conn, _make_prediction(ticker="MSFT"))
        assert id2 > id1

    def test_save_with_feature_snapshot_id(self) -> None:
        conn = _setup_db()
        pred = _make_prediction()
        row_id = save_prediction(conn, pred, feature_snapshot_id=42)
        loaded = load_prediction(conn, row_id)
        assert loaded is not None
        assert loaded.feature_snapshot_id == 42

    def test_save_without_feature_snapshot_id(self) -> None:
        conn = _setup_db()
        pred = _make_prediction()
        row_id = save_prediction(conn, pred)
        loaded = load_prediction(conn, row_id)
        assert loaded is not None
        assert loaded.feature_snapshot_id is None


class TestLoadPrediction:
    def test_round_trip(self) -> None:
        """Core AC: save and load prediction, verify all fields match."""
        conn = _setup_db()
        pred = _make_prediction(
            ticker="GOOG",
            date=datetime.date(2025, 3, 10),
            direction=Direction.DOWN,
            confidence=0.62,
            model_version="v3",
        )
        row_id = save_prediction(conn, pred, feature_snapshot_id=7)
        loaded = load_prediction(conn, row_id)

        assert loaded is not None
        assert isinstance(loaded, PredictionRecord)
        assert loaded.id == row_id
        assert loaded.ticker == "GOOG"
        assert loaded.prediction_date == datetime.date(2025, 3, 10)
        assert loaded.direction == Direction.DOWN
        assert loaded.confidence == 0.62
        assert loaded.model_version == "v3"
        assert loaded.feature_snapshot_id == 7
        assert loaded.created_at  # non-empty ISO datetime string

    def test_load_nonexistent_returns_none(self) -> None:
        conn = _setup_db()
        assert load_prediction(conn, 9999) is None

    def test_direction_up_round_trip(self) -> None:
        conn = _setup_db()
        pred = _make_prediction(direction=Direction.UP)
        row_id = save_prediction(conn, pred)
        loaded = load_prediction(conn, row_id)
        assert loaded is not None
        assert loaded.direction == Direction.UP

    def test_direction_down_round_trip(self) -> None:
        conn = _setup_db()
        pred = _make_prediction(direction=Direction.DOWN)
        row_id = save_prediction(conn, pred)
        loaded = load_prediction(conn, row_id)
        assert loaded is not None
        assert loaded.direction == Direction.DOWN

    def test_created_at_is_iso_datetime(self) -> None:
        conn = _setup_db()
        pred = _make_prediction()
        row_id = save_prediction(conn, pred)
        loaded = load_prediction(conn, row_id)
        assert loaded is not None
        # Verify created_at parses as ISO datetime
        dt = datetime.datetime.fromisoformat(loaded.created_at)
        assert dt.tzinfo is not None  # UTC timezone present
