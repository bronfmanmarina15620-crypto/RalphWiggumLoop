"""Tests for fundamentals table and idempotent upsert (US-107)."""

from __future__ import annotations

import datetime

from smaps.db import ensure_schema, get_connection, upsert_fundamentals
from smaps.models import Fundamentals


def _make_fundamentals(
    ticker: str = "AAPL",
    date: datetime.date | None = None,
    pe_ratio: float | None = 28.5,
    market_cap: float | None = 2.8e12,
    eps: float | None = 6.14,
    revenue: float | None = 3.94e11,
) -> Fundamentals:
    """Helper to build a test Fundamentals with sensible defaults."""
    return Fundamentals(
        ticker=ticker,
        date=date or datetime.date(2025, 1, 15),
        pe_ratio=pe_ratio,
        market_cap=market_cap,
        eps=eps,
        revenue=revenue,
    )


def test_upsert_inserts_fundamentals():
    """A single fundamentals row can be inserted and read back."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    f = _make_fundamentals()
    upsert_fundamentals(conn, [f])

    cur = conn.execute(
        "SELECT ticker, date, pe_ratio, market_cap, eps, revenue FROM fundamentals_daily"
    )
    row = cur.fetchone()
    assert row is not None
    assert row[0] == "AAPL"
    assert row[1] == "2025-01-15"
    assert row[2] == 28.5
    assert row[3] == 2.8e12
    assert row[4] == 6.14
    assert row[5] == 3.94e11
    conn.close()


def test_upsert_same_row_twice_yields_single_row():
    """Inserting the same (ticker, date) twice results in one row; second value wins."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    f1 = _make_fundamentals(pe_ratio=25.0)
    f2 = _make_fundamentals(pe_ratio=30.0)  # same PK, different pe_ratio

    upsert_fundamentals(conn, [f1])
    upsert_fundamentals(conn, [f2])

    cur = conn.execute(
        "SELECT COUNT(*) FROM fundamentals_daily "
        "WHERE ticker='AAPL' AND date='2025-01-15'"
    )
    assert cur.fetchone()[0] == 1

    # Verify the second insert replaced the first
    cur = conn.execute(
        "SELECT pe_ratio FROM fundamentals_daily "
        "WHERE ticker='AAPL' AND date='2025-01-15'"
    )
    assert cur.fetchone()[0] == 30.0
    conn.close()


def test_upsert_multiple_fundamentals():
    """Multiple fundamentals for different dates are all inserted."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    rows = [
        _make_fundamentals(date=datetime.date(2025, 1, 15)),
        _make_fundamentals(date=datetime.date(2025, 1, 16)),
        _make_fundamentals(date=datetime.date(2025, 1, 17)),
    ]
    count = upsert_fundamentals(conn, rows)

    assert count == 3
    cur = conn.execute("SELECT COUNT(*) FROM fundamentals_daily")
    assert cur.fetchone()[0] == 3
    conn.close()


def test_upsert_with_none_fields():
    """Fundamentals with None fields are stored as NULL in the database."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    f = _make_fundamentals(pe_ratio=None, market_cap=None, eps=None, revenue=None)
    upsert_fundamentals(conn, [f])

    cur = conn.execute(
        "SELECT pe_ratio, market_cap, eps, revenue FROM fundamentals_daily"
    )
    row = cur.fetchone()
    assert row is not None
    assert row[0] is None
    assert row[1] is None
    assert row[2] is None
    assert row[3] is None
    conn.close()


def test_upsert_empty_list():
    """Upserting an empty list succeeds without error."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    count = upsert_fundamentals(conn, [])
    assert count == 0

    cur = conn.execute("SELECT COUNT(*) FROM fundamentals_daily")
    assert cur.fetchone()[0] == 0
    conn.close()
