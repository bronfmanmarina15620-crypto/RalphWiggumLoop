"""Tests for FundamentalFeatures (US-204)."""

from __future__ import annotations

import datetime
import math

import pytest

from smaps.db import ensure_schema, get_connection, upsert_fundamentals
from smaps.features.fundamental import FundamentalFeatures
from smaps.models import Fundamentals

EXPECTED_KEYS = {"pe_ratio", "eps", "market_cap"}


def _make_fundamentals(
    ticker: str,
    date: datetime.date,
    pe_ratio: float | None = 25.0,
    eps: float | None = 3.5,
    market_cap: float | None = 2.0e12,
    revenue: float | None = 400.0e9,
) -> Fundamentals:
    """Create a Fundamentals object with sensible defaults."""
    return Fundamentals(
        ticker=ticker,
        date=date,
        pe_ratio=pe_ratio,
        market_cap=market_cap,
        eps=eps,
        revenue=revenue,
    )


def _setup(rows: list[Fundamentals]) -> FundamentalFeatures:
    """Insert fundamentals into an in-memory DB and return a FundamentalFeatures instance."""
    conn = get_connection()
    ensure_schema(conn)
    upsert_fundamentals(conn, rows)
    return FundamentalFeatures(conn)


# ── Output shape ────────────────────────────────────────────────────


def test_output_keys_and_types() -> None:
    """transform() returns all expected feature keys as floats."""
    rows = [_make_fundamentals("AAPL", datetime.date(2025, 1, 1))]
    pipeline = _setup(rows)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 1))

    assert set(result.keys()) == EXPECTED_KEYS
    for key, value in result.items():
        assert isinstance(value, float), f"{key} is not float"


# ── Feature values ──────────────────────────────────────────────────


def test_pe_ratio_value() -> None:
    """pe_ratio reflects the latest available value."""
    rows = [_make_fundamentals("AAPL", datetime.date(2025, 1, 1), pe_ratio=30.0)]
    pipeline = _setup(rows)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 1))

    assert result["pe_ratio"] == pytest.approx(30.0)


def test_eps_value() -> None:
    """eps reflects the latest available value."""
    rows = [_make_fundamentals("AAPL", datetime.date(2025, 1, 1), eps=5.25)]
    pipeline = _setup(rows)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 1))

    assert result["eps"] == pytest.approx(5.25)


def test_market_cap_value() -> None:
    """market_cap reflects the latest available value."""
    rows = [_make_fundamentals("AAPL", datetime.date(2025, 1, 1), market_cap=3.0e12)]
    pipeline = _setup(rows)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 1))

    assert result["market_cap"] == pytest.approx(3.0e12)


def test_latest_row_used() -> None:
    """When multiple rows exist, the most recent one <= as_of_date is used."""
    rows = [
        _make_fundamentals("AAPL", datetime.date(2025, 1, 1), pe_ratio=20.0),
        _make_fundamentals("AAPL", datetime.date(2025, 1, 5), pe_ratio=25.0),
        _make_fundamentals("AAPL", datetime.date(2025, 1, 10), pe_ratio=30.0),
    ]
    pipeline = _setup(rows)

    # as_of_date is Jan 7 — should pick the Jan 5 row
    result = pipeline.transform("AAPL", datetime.date(2025, 1, 7))

    assert result["pe_ratio"] == pytest.approx(25.0)


# ── Graceful NaN for missing fields ────────────────────────────────


def test_none_fields_return_nan() -> None:
    """Individual None fields are returned as NaN."""
    rows = [_make_fundamentals("AAPL", datetime.date(2025, 1, 1), pe_ratio=None, eps=None, market_cap=1.0e12)]
    pipeline = _setup(rows)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 1))

    assert math.isnan(result["pe_ratio"])
    assert math.isnan(result["eps"])
    assert result["market_cap"] == pytest.approx(1.0e12)


def test_no_data_returns_all_nan() -> None:
    """All features return NaN when no fundamentals data is available."""
    conn = get_connection()
    ensure_schema(conn)
    pipeline = FundamentalFeatures(conn)

    result = pipeline.transform("FAKE", datetime.date(2025, 1, 1))

    for key in EXPECTED_KEYS:
        assert math.isnan(result[key]), f"{key} should be NaN"


def test_no_data_for_ticker_but_other_tickers_exist() -> None:
    """Returns NaN for a ticker with no data even when others have data."""
    rows = [_make_fundamentals("MSFT", datetime.date(2025, 1, 1))]
    pipeline = _setup(rows)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 1))

    for key in EXPECTED_KEYS:
        assert math.isnan(result[key]), f"{key} should be NaN"
