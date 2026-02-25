"""Tests for TechnicalFeatures (US-202)."""

from __future__ import annotations

import datetime
import math

import pytest

from smaps.db import ensure_schema, get_connection, upsert_bars
from smaps.features.technical import TechnicalFeatures
from smaps.models import OHLCVBar

EXPECTED_KEYS = {
    "return_1d",
    "return_5d",
    "return_10d",
    "ma_ratio_5_20",
    "volume_change_1d",
    "volatility_20d",
    "rsi_14",
}


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


def _setup(bars: list[OHLCVBar]) -> TechnicalFeatures:
    """Insert bars into an in-memory DB and return a TechnicalFeatures instance."""
    conn = get_connection()
    ensure_schema(conn)
    upsert_bars(conn, bars)
    return TechnicalFeatures(conn)


# ── Output shape ────────────────────────────────────────────────────

def test_output_keys_and_shape() -> None:
    """transform() returns all expected feature keys as floats."""
    closes = [100.0 + i for i in range(25)]
    bars = _make_bars("AAPL", datetime.date(2025, 1, 1), closes)
    pipeline = _setup(bars)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 25))

    assert set(result.keys()) == EXPECTED_KEYS
    for key, value in result.items():
        assert isinstance(value, float), f"{key} is not float"


# ── Individual indicators ───────────────────────────────────────────

def test_return_1d() -> None:
    """return_1d = close[-1] / close[-2] - 1."""
    closes = [100.0, 110.0]
    bars = _make_bars("AAPL", datetime.date(2025, 1, 1), closes)
    pipeline = _setup(bars)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 2))

    assert result["return_1d"] == pytest.approx(0.1)


def test_return_5d() -> None:
    """return_5d = close[-1] / close[-6] - 1."""
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 120.0]
    bars = _make_bars("AAPL", datetime.date(2025, 1, 1), closes)
    pipeline = _setup(bars)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 6))

    assert result["return_5d"] == pytest.approx(0.2)


def test_return_10d() -> None:
    """return_10d = close[-1] / close[-11] - 1."""
    closes = [50.0] + [50.0] * 9 + [75.0]  # 11 bars total
    bars = _make_bars("AAPL", datetime.date(2025, 1, 1), closes)
    pipeline = _setup(bars)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 11))

    assert result["return_10d"] == pytest.approx(0.5)


def test_volume_change_1d() -> None:
    """volume_change_1d = volume[-1] / volume[-2] - 1."""
    closes = [100.0, 100.0]
    volumes = [1000, 1500]
    bars = _make_bars("AAPL", datetime.date(2025, 1, 1), closes, volumes)
    pipeline = _setup(bars)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 2))

    assert result["volume_change_1d"] == pytest.approx(0.5)


def test_rsi_all_gains() -> None:
    """RSI is 100 when all changes are positive."""
    closes = [100.0 + i for i in range(16)]  # 15 consecutive gains
    bars = _make_bars("AAPL", datetime.date(2025, 1, 1), closes)
    pipeline = _setup(bars)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 16))

    assert result["rsi_14"] == 100.0


def test_rsi_all_losses() -> None:
    """RSI is 0 when all changes are negative."""
    closes = [200.0 - i for i in range(16)]  # 15 consecutive losses
    bars = _make_bars("AAPL", datetime.date(2025, 1, 1), closes)
    pipeline = _setup(bars)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 16))

    assert result["rsi_14"] == pytest.approx(0.0)


def test_ma_ratio_5_20() -> None:
    """ma_ratio_5_20 = MA(5) / MA(20) when enough data is available."""
    closes = [100.0] * 15 + [200.0] * 5  # MA(5)=200, MA(20)=125
    bars = _make_bars("AAPL", datetime.date(2025, 1, 1), closes)
    pipeline = _setup(bars)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 20))

    assert result["ma_ratio_5_20"] == pytest.approx(200.0 / 125.0)


# ── No future data leakage ─────────────────────────────────────────

def test_no_leakage() -> None:
    """Features only use bars dated <= as_of_date."""
    closes = [100.0, 110.0, 90.0]
    bars = _make_bars("AAPL", datetime.date(2025, 1, 1), closes)
    pipeline = _setup(bars)

    # as_of_date = day 2 → bar on 2025-01-03 (close=90) excluded
    result = pipeline.transform("AAPL", datetime.date(2025, 1, 2))

    assert result["return_1d"] == pytest.approx(0.1)


# ── Edge cases ──────────────────────────────────────────────────────

def test_insufficient_data_returns_nan() -> None:
    """All features return NaN when only 1 bar is available."""
    closes = [100.0]
    bars = _make_bars("AAPL", datetime.date(2025, 1, 1), closes)
    pipeline = _setup(bars)

    result = pipeline.transform("AAPL", datetime.date(2025, 1, 1))

    assert set(result.keys()) == EXPECTED_KEYS
    for key in EXPECTED_KEYS:
        assert math.isnan(result[key]), f"{key} should be NaN with 1 bar"


def test_no_bars_returns_nan() -> None:
    """All features return NaN for an unknown ticker."""
    conn = get_connection()
    ensure_schema(conn)
    pipeline = TechnicalFeatures(conn)

    result = pipeline.transform("FAKE", datetime.date(2025, 1, 1))

    assert set(result.keys()) == EXPECTED_KEYS
    for key in EXPECTED_KEYS:
        assert math.isnan(result[key]), f"{key} should be NaN with no data"
