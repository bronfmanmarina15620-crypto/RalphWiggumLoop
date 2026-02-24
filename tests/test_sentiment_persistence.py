"""Tests for sentiment table and idempotent upsert (US-105)."""

from __future__ import annotations

import datetime

from smaps.db import ensure_schema, get_connection, upsert_sentiment
from smaps.models import SentimentScore


def _make_score(
    ticker: str = "AAPL",
    date: datetime.date | None = None,
    score: float = 0.5,
    source: str = "google_news_rss",
) -> SentimentScore:
    """Helper to build a test sentiment score with sensible defaults."""
    return SentimentScore(
        ticker=ticker,
        date=date or datetime.date(2025, 1, 15),
        score=score,
        source=source,
    )


def test_upsert_inserts_sentiment():
    """A single sentiment score can be inserted and read back."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    s = _make_score()
    upsert_sentiment(conn, [s])

    cur = conn.execute("SELECT ticker, date, score, source FROM sentiment_daily")
    row = cur.fetchone()
    assert row is not None
    assert row[0] == "AAPL"
    assert row[1] == "2025-01-15"
    assert row[2] == 0.5
    assert row[3] == "google_news_rss"
    conn.close()


def test_upsert_same_row_twice_yields_single_row():
    """Inserting the same (ticker, date, source) twice results in one row."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    s1 = _make_score(score=0.3)
    s2 = _make_score(score=0.8)  # same PK, different score

    upsert_sentiment(conn, [s1])
    upsert_sentiment(conn, [s2])

    cur = conn.execute(
        "SELECT COUNT(*) FROM sentiment_daily "
        "WHERE ticker='AAPL' AND date='2025-01-15' AND source='google_news_rss'"
    )
    assert cur.fetchone()[0] == 1

    # Verify the second insert replaced the first
    cur = conn.execute(
        "SELECT score FROM sentiment_daily "
        "WHERE ticker='AAPL' AND date='2025-01-15' AND source='google_news_rss'"
    )
    assert cur.fetchone()[0] == 0.8
    conn.close()


def test_upsert_multiple_scores():
    """Multiple sentiment scores for different dates are all inserted."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    scores = [
        _make_score(date=datetime.date(2025, 1, 15)),
        _make_score(date=datetime.date(2025, 1, 16)),
        _make_score(date=datetime.date(2025, 1, 17)),
    ]
    count = upsert_sentiment(conn, scores)

    assert count == 3
    cur = conn.execute("SELECT COUNT(*) FROM sentiment_daily")
    assert cur.fetchone()[0] == 3
    conn.close()


def test_upsert_different_sources_same_ticker_date():
    """Different sources for the same (ticker, date) coexist as separate rows."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    s1 = _make_score(source="google_news_rss", score=0.5)
    s2 = _make_score(source="reddit", score=-0.3)

    upsert_sentiment(conn, [s1, s2])

    cur = conn.execute(
        "SELECT COUNT(*) FROM sentiment_daily WHERE ticker='AAPL' AND date='2025-01-15'"
    )
    assert cur.fetchone()[0] == 2
    conn.close()


def test_upsert_empty_list():
    """Upserting an empty list succeeds without error."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    count = upsert_sentiment(conn, [])
    assert count == 0

    cur = conn.execute("SELECT COUNT(*) FROM sentiment_daily")
    assert cur.fetchone()[0] == 0
    conn.close()
