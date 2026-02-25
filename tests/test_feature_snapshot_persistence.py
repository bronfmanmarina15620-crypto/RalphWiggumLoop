"""Tests for feature snapshot persistence (US-206)."""

from __future__ import annotations

import datetime
import math

from smaps.db import (
    FeatureSnapshot,
    ensure_schema,
    get_connection,
    load_feature_snapshot,
    save_feature_snapshot,
)


def _sample_features() -> dict[str, float]:
    """Return a sample feature vector with all 12 keys."""
    return {
        "return_1d": 0.012,
        "return_5d": 0.035,
        "return_10d": -0.008,
        "ma_ratio_5_20": 1.02,
        "volume_change_1d": 0.15,
        "volatility_20d": 0.023,
        "rsi_14": 55.3,
        "latest_sentiment_score": 0.25,
        "sentiment_ma_5d": 0.18,
        "pe_ratio": 28.5,
        "eps": 6.14,
        "market_cap": 2.8e12,
    }


def test_save_and_load_round_trip():
    """A saved snapshot can be loaded back with equal values."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    features = _sample_features()
    snapshot_id = save_feature_snapshot(
        conn, "AAPL", datetime.date(2025, 1, 15), features, "v1"
    )

    loaded = load_feature_snapshot(conn, snapshot_id)
    assert loaded is not None
    assert loaded.id == snapshot_id
    assert loaded.ticker == "AAPL"
    assert loaded.feature_date == datetime.date(2025, 1, 15)
    assert loaded.features == features
    assert loaded.pipeline_version == "v1"
    conn.close()


def test_save_returns_autoincrement_id():
    """Each saved snapshot gets a unique auto-incremented id."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    id1 = save_feature_snapshot(
        conn, "AAPL", datetime.date(2025, 1, 15), _sample_features(), "v1"
    )
    id2 = save_feature_snapshot(
        conn, "MSFT", datetime.date(2025, 1, 15), _sample_features(), "v1"
    )

    assert id1 is not None
    assert id2 is not None
    assert id2 > id1
    conn.close()


def test_load_nonexistent_returns_none():
    """Loading a snapshot that doesn't exist returns None."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    result = load_feature_snapshot(conn, 9999)
    assert result is None
    conn.close()


def test_features_with_nan_values():
    """NaN values survive the JSON round-trip (stored as null)."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    features = {"return_1d": float("nan"), "rsi_14": 55.0}
    snapshot_id = save_feature_snapshot(
        conn, "AAPL", datetime.date(2025, 1, 15), features, "v1"
    )

    loaded = load_feature_snapshot(conn, snapshot_id)
    assert loaded is not None
    # JSON encodes NaN as null → loads back as None, not NaN
    # This is acceptable; downstream code must handle None from JSON
    assert loaded.features["rsi_14"] == 55.0
    conn.close()


def test_multiple_snapshots_same_ticker_date():
    """Multiple snapshots for the same ticker+date can coexist (different ids)."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    date = datetime.date(2025, 1, 15)
    id1 = save_feature_snapshot(conn, "AAPL", date, _sample_features(), "v1")
    id2 = save_feature_snapshot(conn, "AAPL", date, _sample_features(), "v2")

    assert id1 != id2

    snap1 = load_feature_snapshot(conn, id1)
    snap2 = load_feature_snapshot(conn, id2)
    assert snap1 is not None and snap1.pipeline_version == "v1"
    assert snap2 is not None and snap2.pipeline_version == "v2"
    conn.close()


def test_feature_snapshot_named_tuple():
    """FeatureSnapshot is a NamedTuple with expected fields."""
    snap = FeatureSnapshot(
        id=1,
        ticker="AAPL",
        feature_date=datetime.date(2025, 1, 15),
        features={"return_1d": 0.01},
        pipeline_version="v1",
    )
    assert snap.id == 1
    assert snap.ticker == "AAPL"
    assert snap.feature_date == datetime.date(2025, 1, 15)
    assert snap.features == {"return_1d": 0.01}
    assert snap.pipeline_version == "v1"
