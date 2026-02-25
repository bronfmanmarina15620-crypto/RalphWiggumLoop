"""Tests for SentimentFeatures (US-203)."""

from __future__ import annotations

import datetime

import pytest

from smaps.db import ensure_schema, get_connection, upsert_sentiment
from smaps.features.sentiment import SentimentFeatures
from smaps.models import SentimentScore

EXPECTED_KEYS = {"latest_sentiment_score", "sentiment_ma_5d"}


def _make_scores(
    ticker: str,
    start: datetime.date,
    values: list[float],
    source: str = "google_news",
) -> list[SentimentScore]:
    """Create synthetic SentimentScore objects with given score values."""
    return [
        SentimentScore(
            ticker=ticker,
            date=start + datetime.timedelta(days=i),
            score=v,
            source=source,
        )
        for i, v in enumerate(values)
    ]


def _setup(scores: list[SentimentScore]) -> SentimentFeatures:
    """Insert scores into an in-memory DB and return a SentimentFeatures instance."""
    conn = get_connection()
    ensure_schema(conn)
    upsert_sentiment(conn, scores)
    return SentimentFeatures(conn)


# ── Output shape ────────────────────────────────────────────────────


def test_output_keys_and_types() -> None:
    """transform() returns all expected feature keys as floats."""
    scores = _make_scores("AAPL", datetime.date(2025, 1, 1), [0.5, 0.3, 0.1])
    pipeline = _setup(scores)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 3))

    assert set(result.keys()) == EXPECTED_KEYS
    for key, value in result.items():
        assert isinstance(value, float), f"{key} is not float"


# ── latest_sentiment_score ──────────────────────────────────────────


def test_latest_sentiment_score() -> None:
    """latest_sentiment_score is the most recent score <= as_of_date."""
    scores = _make_scores("AAPL", datetime.date(2025, 1, 1), [0.2, 0.5, 0.8])
    pipeline = _setup(scores)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 3))

    assert result["latest_sentiment_score"] == pytest.approx(0.8)


def test_latest_sentiment_score_respects_as_of_date() -> None:
    """Future scores are excluded by as_of_date filter."""
    scores = _make_scores("AAPL", datetime.date(2025, 1, 1), [0.2, 0.5, 0.8])
    pipeline = _setup(scores)

    # Only scores on Jan 1 and Jan 2 are visible
    result = pipeline.transform("AAPL", datetime.date(2025, 1, 2))

    assert result["latest_sentiment_score"] == pytest.approx(0.5)


# ── sentiment_ma_5d ─────────────────────────────────────────────────


def test_sentiment_ma_5d_with_enough_data() -> None:
    """sentiment_ma_5d averages the last 5 scores."""
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    scores = _make_scores("AAPL", datetime.date(2025, 1, 1), values)
    pipeline = _setup(scores)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 7))

    # Last 5: [0.3, 0.4, 0.5, 0.6, 0.7] → mean = 0.5
    assert result["sentiment_ma_5d"] == pytest.approx(0.5)


def test_sentiment_ma_5d_with_fewer_than_5() -> None:
    """sentiment_ma_5d averages all available scores when fewer than 5."""
    values = [0.2, 0.4]
    scores = _make_scores("AAPL", datetime.date(2025, 1, 1), values)
    pipeline = _setup(scores)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 2))

    # Only 2 scores available → mean of [0.2, 0.4] = 0.3
    assert result["sentiment_ma_5d"] == pytest.approx(0.3)


# ── Graceful fallback to 0.0 ────────────────────────────────────────


def test_no_data_returns_zeros() -> None:
    """All features return 0.0 when no sentiment data is available."""
    conn = get_connection()
    ensure_schema(conn)
    pipeline = SentimentFeatures(conn)

    result = pipeline.transform("FAKE", datetime.date(2025, 1, 1))

    assert result["latest_sentiment_score"] == 0.0
    assert result["sentiment_ma_5d"] == 0.0


def test_no_data_for_ticker_but_other_tickers_exist() -> None:
    """Returns 0.0 for a ticker with no data even when others have data."""
    scores = _make_scores("MSFT", datetime.date(2025, 1, 1), [0.5])
    pipeline = _setup(scores)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 1))

    assert result["latest_sentiment_score"] == 0.0
    assert result["sentiment_ma_5d"] == 0.0
