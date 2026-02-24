"""Tests for OHLCV table and idempotent upsert (US-103)."""

from __future__ import annotations

import datetime

from smaps.db import ensure_schema, get_connection, upsert_bars
from smaps.models import OHLCVBar


def _make_bar(
    ticker: str = "AAPL",
    date: datetime.date | None = None,
    close: float = 150.0,
) -> OHLCVBar:
    """Helper to build a test bar with sensible defaults."""
    return OHLCVBar(
        ticker=ticker,
        date=date or datetime.date(2025, 1, 15),
        open=148.0,
        high=151.0,
        low=147.0,
        close=close,
        volume=1_000_000,
    )


def test_upsert_inserts_bar():
    """A single bar can be inserted and read back."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    bar = _make_bar()
    upsert_bars(conn, [bar])

    cur = conn.execute("SELECT ticker, date, open, high, low, close, volume FROM ohlcv_daily")
    row = cur.fetchone()
    assert row is not None
    assert row[0] == "AAPL"
    assert row[1] == "2025-01-15"
    assert row[5] == 150.0
    assert row[6] == 1_000_000
    conn.close()


def test_upsert_same_row_twice_yields_single_row():
    """Inserting the same (ticker, date) twice results in one row (idempotent)."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    bar_v1 = _make_bar(close=150.0)
    bar_v2 = _make_bar(close=155.0)  # same PK, different close

    upsert_bars(conn, [bar_v1])
    upsert_bars(conn, [bar_v2])

    cur = conn.execute("SELECT COUNT(*) FROM ohlcv_daily WHERE ticker='AAPL' AND date='2025-01-15'")
    assert cur.fetchone()[0] == 1

    # Verify the second insert replaced the first
    cur = conn.execute("SELECT close FROM ohlcv_daily WHERE ticker='AAPL' AND date='2025-01-15'")
    assert cur.fetchone()[0] == 155.0
    conn.close()


def test_upsert_multiple_bars():
    """Multiple bars for different dates are all inserted."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    bars = [
        _make_bar(date=datetime.date(2025, 1, 15)),
        _make_bar(date=datetime.date(2025, 1, 16)),
        _make_bar(date=datetime.date(2025, 1, 17)),
    ]
    count = upsert_bars(conn, bars)

    assert count == 3
    cur = conn.execute("SELECT COUNT(*) FROM ohlcv_daily")
    assert cur.fetchone()[0] == 3
    conn.close()


def test_upsert_empty_list():
    """Upserting an empty list succeeds without error."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    count = upsert_bars(conn, [])
    assert count == 0

    cur = conn.execute("SELECT COUNT(*) FROM ohlcv_daily")
    assert cur.fetchone()[0] == 0
    conn.close()
