"""Leakage prevention tests (US-207).

Verify that no future data leaks into features:
- Injecting a known future bar does not influence features computed at date T.
- Feature snapshot feature_date is always <= the as_of_date used to build features.
"""

from __future__ import annotations

import datetime
import math

import pytest

from smaps.db import (
    ensure_schema,
    get_connection,
    load_feature_snapshot,
    save_feature_snapshot,
    upsert_bars,
    upsert_fundamentals,
    upsert_sentiment,
)
from smaps.features.combined import build_features
from smaps.features.fundamental import FundamentalFeatures
from smaps.features.sentiment import SentimentFeatures
from smaps.features.technical import TechnicalFeatures
from smaps.models import Fundamentals, OHLCVBar, SentimentScore


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
                low=max(c - 1.0, 0.01),
                close=c,
                volume=v,
            )
        )
    return bars


def _fresh_db() -> "import('sqlite3').Connection":
    """Create an in-memory DB with schema applied."""
    import sqlite3

    conn = get_connection(":memory:")
    ensure_schema(conn)
    return conn


# ── AC-1: Inject known future bar, verify it does not appear in features ──


class TestFutureBarExcluded:
    """Across all pipelines, data dated after as_of_date must be invisible."""

    AS_OF = datetime.date(2025, 1, 20)

    def test_technical_excludes_future_bar(self) -> None:
        """A future OHLCV bar (after as_of_date) must NOT affect technical features."""
        conn = _fresh_db()

        # 20 bars from Jan 1..Jan 20 with close = 100
        historical_bars = _make_bars("AAPL", datetime.date(2025, 1, 1), [100.0] * 20)
        upsert_bars(conn, historical_bars)

        # Compute features WITHOUT the future bar
        pipeline = TechnicalFeatures(conn)
        features_before = pipeline.transform("AAPL", self.AS_OF)

        # Inject a wildly different future bar on Jan 21 (close=999)
        future_bar = OHLCVBar(
            ticker="AAPL",
            date=datetime.date(2025, 1, 21),
            open=999.0,
            high=1000.0,
            low=998.0,
            close=999.0,
            volume=999_999,
        )
        upsert_bars(conn, [future_bar])

        # Compute features again at the SAME as_of_date
        features_after = pipeline.transform("AAPL", self.AS_OF)

        # Features must be identical — the future bar must be invisible
        for key in features_before:
            before = features_before[key]
            after = features_after[key]
            if math.isnan(before):
                assert math.isnan(after), f"{key}: expected NaN, got {after}"
            else:
                assert before == pytest.approx(after), (
                    f"{key}: changed from {before} to {after} after injecting future bar"
                )

    def test_sentiment_excludes_future_score(self) -> None:
        """A future sentiment score (after as_of_date) must NOT affect sentiment features."""
        conn = _fresh_db()

        # 5 historical sentiment scores
        historical_scores = [
            SentimentScore(
                ticker="AAPL",
                date=datetime.date(2025, 1, 16 + i),
                score=0.2,
                source="google_news",
            )
            for i in range(5)
        ]
        upsert_sentiment(conn, historical_scores)

        pipeline = SentimentFeatures(conn)
        features_before = pipeline.transform("AAPL", self.AS_OF)

        # Inject a future sentiment score with a very different value
        future_score = SentimentScore(
            ticker="AAPL",
            date=datetime.date(2025, 1, 21),
            score=-0.9,
            source="google_news",
        )
        upsert_sentiment(conn, [future_score])

        features_after = pipeline.transform("AAPL", self.AS_OF)

        for key in features_before:
            assert features_before[key] == pytest.approx(features_after[key]), (
                f"{key}: changed from {features_before[key]} to {features_after[key]} "
                "after injecting future sentiment"
            )

    def test_fundamental_excludes_future_row(self) -> None:
        """A future fundamentals row (after as_of_date) must NOT affect fundamental features."""
        conn = _fresh_db()

        historical_fund = Fundamentals(
            ticker="AAPL",
            date=datetime.date(2025, 1, 15),
            pe_ratio=25.0,
            market_cap=3e12,
            eps=6.5,
            revenue=4e11,
        )
        upsert_fundamentals(conn, [historical_fund])

        pipeline = FundamentalFeatures(conn)
        features_before = pipeline.transform("AAPL", self.AS_OF)

        # Inject a future fundamentals row with very different values
        future_fund = Fundamentals(
            ticker="AAPL",
            date=datetime.date(2025, 1, 21),
            pe_ratio=999.0,
            market_cap=1e15,
            eps=999.0,
            revenue=1e15,
        )
        upsert_fundamentals(conn, [future_fund])

        features_after = pipeline.transform("AAPL", self.AS_OF)

        for key in features_before:
            before = features_before[key]
            after = features_after[key]
            if math.isnan(before):
                assert math.isnan(after), f"{key}: expected NaN, got {after}"
            else:
                assert before == pytest.approx(after), (
                    f"{key}: changed from {before} to {after} after injecting future fundamentals"
                )

    def test_combined_features_exclude_all_future_data(self) -> None:
        """build_features is unaffected by future data across all three pipelines."""
        conn = _fresh_db()

        # Populate historical data for all three sources
        bars = _make_bars("AAPL", datetime.date(2025, 1, 1), [100.0 + i for i in range(20)])
        upsert_bars(conn, bars)

        scores = [
            SentimentScore(
                ticker="AAPL",
                date=datetime.date(2025, 1, 16 + i),
                score=0.3,
                source="google_news",
            )
            for i in range(5)
        ]
        upsert_sentiment(conn, scores)

        fund = Fundamentals(
            ticker="AAPL",
            date=datetime.date(2025, 1, 15),
            pe_ratio=30.0,
            market_cap=2.5e12,
            eps=7.0,
            revenue=3.5e11,
        )
        upsert_fundamentals(conn, [fund])

        features_before = build_features(conn, "AAPL", self.AS_OF)

        # Inject future data across ALL sources
        future_bar = OHLCVBar(
            ticker="AAPL",
            date=datetime.date(2025, 1, 21),
            open=500.0,
            high=501.0,
            low=499.0,
            close=500.0,
            volume=500_000,
        )
        upsert_bars(conn, [future_bar])

        future_score = SentimentScore(
            ticker="AAPL",
            date=datetime.date(2025, 1, 21),
            score=-1.0,
            source="google_news",
        )
        upsert_sentiment(conn, [future_score])

        future_fund = Fundamentals(
            ticker="AAPL",
            date=datetime.date(2025, 1, 21),
            pe_ratio=1.0,
            market_cap=1.0,
            eps=0.01,
            revenue=1.0,
        )
        upsert_fundamentals(conn, [future_fund])

        features_after = build_features(conn, "AAPL", self.AS_OF)

        for key in features_before:
            before = features_before[key]
            after = features_after[key]
            if math.isnan(before):
                assert math.isnan(after), f"{key}: expected NaN, got {after}"
            else:
                assert before == pytest.approx(after), (
                    f"{key}: changed from {before} to {after} after injecting future data"
                )


# ── AC-2: feature_date in snapshot <= as_of_date ─────────────────────


class TestSnapshotDateConstraint:
    """Feature snapshots must have feature_date <= as_of_date."""

    def test_snapshot_feature_date_equals_as_of_date(self) -> None:
        """When we build features and save a snapshot, feature_date == as_of_date."""
        conn = _fresh_db()

        as_of = datetime.date(2025, 1, 15)
        features = build_features(conn, "AAPL", as_of)
        snapshot_id = save_feature_snapshot(conn, "AAPL", as_of, features, "v1")
        snapshot = load_feature_snapshot(conn, snapshot_id)

        assert snapshot is not None
        assert snapshot.feature_date <= as_of

    def test_snapshot_feature_date_not_in_future(self) -> None:
        """A saved snapshot's feature_date must not exceed the as_of_date used."""
        conn = _fresh_db()

        # Build features for multiple dates and verify each snapshot
        for day_offset in range(1, 6):
            as_of = datetime.date(2025, 1, 10 + day_offset)
            features = build_features(conn, "AAPL", as_of)
            snapshot_id = save_feature_snapshot(
                conn, "AAPL", as_of, features, "v1"
            )
            snapshot = load_feature_snapshot(conn, snapshot_id)

            assert snapshot is not None
            assert snapshot.feature_date <= as_of, (
                f"snapshot feature_date {snapshot.feature_date} > as_of_date {as_of}"
            )

    def test_snapshot_preserves_features_at_point_in_time(self) -> None:
        """A snapshot taken at date T should match features recomputed at date T,
        even after new data is added for dates after T."""
        conn = _fresh_db()

        as_of = datetime.date(2025, 1, 20)
        bars = _make_bars("AAPL", datetime.date(2025, 1, 1), [100.0] * 20)
        upsert_bars(conn, bars)

        # Build and save snapshot at T
        features_at_t = build_features(conn, "AAPL", as_of)
        snapshot_id = save_feature_snapshot(conn, "AAPL", as_of, features_at_t, "v1")

        # Add future data
        future_bars = _make_bars("AAPL", datetime.date(2025, 1, 21), [200.0] * 5)
        upsert_bars(conn, future_bars)

        # Load snapshot — should still reflect the original features
        snapshot = load_feature_snapshot(conn, snapshot_id)
        assert snapshot is not None
        assert snapshot.feature_date <= as_of

        for key in features_at_t:
            saved = snapshot.features.get(key)
            original = features_at_t[key]
            if saved is None:
                # JSON null for NaN values
                assert math.isnan(original), f"{key}: snapshot null but original was {original}"
            elif math.isnan(original):
                # NaN encoded as null in JSON
                assert saved is None or (isinstance(saved, float) and math.isnan(saved))
            else:
                assert saved == pytest.approx(original), (
                    f"{key}: snapshot has {saved}, expected {original}"
                )
