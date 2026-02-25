"""Tests for build_features (US-205)."""

from __future__ import annotations

import datetime
import math

from smaps.db import (
    ensure_schema,
    get_connection,
    upsert_bars,
    upsert_fundamentals,
    upsert_sentiment,
)
from smaps.features.combined import FEATURE_KEYS, build_features
from smaps.models import Fundamentals, OHLCVBar, SentimentScore


def _setup_db() -> "tuple[__import__('sqlite3').Connection]":
    """Create an in-memory DB with schema applied."""
    import sqlite3

    conn = get_connection(":memory:")
    ensure_schema(conn)
    return (conn,)


def _make_bars(
    ticker: str,
    start: datetime.date,
    closes: list[float],
    volumes: list[int] | None = None,
) -> list[OHLCVBar]:
    """Create synthetic OHLCV bars with given close prices."""
    if volumes is None:
        volumes = [1000] * len(closes)
    bars: list[OHLCVBar] = []
    for i, (c, v) in enumerate(zip(closes, volumes)):
        d = start + datetime.timedelta(days=i)
        bars.append(
            OHLCVBar(
                ticker=ticker,
                date=d,
                open=c,
                high=c + 1.0,
                low=c - 1.0,
                close=c,
                volume=v,
            )
        )
    return bars


class TestBuildFeatures:
    """Tests for the unified build_features function."""

    def test_returns_all_expected_keys(self) -> None:
        """build_features returns exactly the canonical key set."""
        (conn,) = _setup_db()
        # No data at all — should still return all keys
        result = build_features(conn, "AAPL", datetime.date(2025, 1, 15))
        assert set(result.keys()) == FEATURE_KEYS

    def test_all_values_are_floats(self) -> None:
        """Every value in the feature vector is a float."""
        (conn,) = _setup_db()
        result = build_features(conn, "AAPL", datetime.date(2025, 1, 15))
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is {type(value)}, expected float"

    def test_consistent_keys_regardless_of_data(self) -> None:
        """Key set is the same whether data exists or not."""
        (conn,) = _setup_db()

        # With no data
        result_empty = build_features(conn, "AAPL", datetime.date(2025, 1, 15))

        # Populate some price data
        bars = _make_bars("AAPL", datetime.date(2025, 1, 1), [100.0 + i for i in range(15)])
        upsert_bars(conn, bars)

        result_with_data = build_features(conn, "AAPL", datetime.date(2025, 1, 15))

        assert set(result_empty.keys()) == set(result_with_data.keys())

    def test_merges_technical_features(self) -> None:
        """Technical features are present in the combined output."""
        (conn,) = _setup_db()
        bars = _make_bars("AAPL", datetime.date(2025, 1, 1), [100.0 + i for i in range(25)])
        upsert_bars(conn, bars)

        result = build_features(conn, "AAPL", datetime.date(2025, 1, 25))
        # return_1d should be a finite float with enough data
        assert math.isfinite(result["return_1d"])
        assert math.isfinite(result["rsi_14"])

    def test_merges_sentiment_features(self) -> None:
        """Sentiment features are present in the combined output."""
        (conn,) = _setup_db()
        scores = [
            SentimentScore(
                ticker="AAPL",
                date=datetime.date(2025, 1, 10 + i),
                score=0.1 * (i + 1),
                source="google_news",
            )
            for i in range(5)
        ]
        upsert_sentiment(conn, scores)

        result = build_features(conn, "AAPL", datetime.date(2025, 1, 15))
        assert result["latest_sentiment_score"] == 0.5  # last score
        assert result["sentiment_ma_5d"] > 0.0

    def test_merges_fundamental_features(self) -> None:
        """Fundamental features are present in the combined output."""
        (conn,) = _setup_db()
        fund = Fundamentals(
            ticker="AAPL",
            date=datetime.date(2025, 1, 10),
            pe_ratio=25.0,
            market_cap=3e12,
            eps=6.5,
            revenue=4e11,
        )
        upsert_fundamentals(conn, [fund])

        result = build_features(conn, "AAPL", datetime.date(2025, 1, 15))
        assert result["pe_ratio"] == 25.0
        assert result["eps"] == 6.5
        assert result["market_cap"] == 3e12

    def test_no_data_defaults(self) -> None:
        """With no data, technical/fundamental return NaN, sentiment returns 0.0."""
        (conn,) = _setup_db()
        result = build_features(conn, "UNKNOWN", datetime.date(2025, 1, 15))

        # Technical features → NaN
        assert math.isnan(result["return_1d"])
        assert math.isnan(result["rsi_14"])
        assert math.isnan(result["volatility_20d"])

        # Sentiment features → 0.0
        assert result["latest_sentiment_score"] == 0.0
        assert result["sentiment_ma_5d"] == 0.0

        # Fundamental features → NaN
        assert math.isnan(result["pe_ratio"])
        assert math.isnan(result["eps"])
        assert math.isnan(result["market_cap"])

    def test_full_data_all_keys_present(self) -> None:
        """With all data sources populated, all features are computed."""
        (conn,) = _setup_db()

        # Price data (25 days for all technical indicators)
        bars = _make_bars("AAPL", datetime.date(2025, 1, 1), [100.0 + i for i in range(25)])
        upsert_bars(conn, bars)

        # Sentiment data
        scores = [
            SentimentScore(
                ticker="AAPL",
                date=datetime.date(2025, 1, 20 + i),
                score=0.3,
                source="google_news",
            )
            for i in range(5)
        ]
        upsert_sentiment(conn, scores)

        # Fundamental data
        fund = Fundamentals(
            ticker="AAPL",
            date=datetime.date(2025, 1, 20),
            pe_ratio=30.0,
            market_cap=2.5e12,
            eps=7.0,
            revenue=3.5e11,
        )
        upsert_fundamentals(conn, [fund])

        result = build_features(conn, "AAPL", datetime.date(2025, 1, 25))

        # All 12 keys present
        assert set(result.keys()) == FEATURE_KEYS
        assert len(result) == 12

        # All values are finite floats
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not float"
            assert math.isfinite(value), f"{key} is not finite: {value}"
