"""Tests for SMAPS data models."""

from __future__ import annotations

import datetime

import pytest

from smaps.models import OHLCVBar


class TestOHLCVBar:
    """Tests for OHLCVBar dataclass."""

    def test_create_valid_bar(self) -> None:
        """Valid OHLCVBar can be created with correct fields."""
        bar = OHLCVBar(
            ticker="AAPL",
            date=datetime.date(2025, 1, 15),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1_000_000,
        )
        assert bar.ticker == "AAPL"
        assert bar.date == datetime.date(2025, 1, 15)
        assert bar.open == 150.0
        assert bar.high == 155.0
        assert bar.low == 149.0
        assert bar.close == 153.0
        assert bar.volume == 1_000_000

    def test_create_bar_high_equals_low(self) -> None:
        """OHLCVBar allows high == low (flat day)."""
        bar = OHLCVBar(
            ticker="MSFT",
            date=datetime.date(2025, 1, 15),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=0,
        )
        assert bar.high == bar.low

    def test_validation_high_less_than_low(self) -> None:
        """OHLCVBar rejects high < low."""
        with pytest.raises(ValueError, match="high.*must be >= low"):
            OHLCVBar(
                ticker="AAPL",
                date=datetime.date(2025, 1, 15),
                open=150.0,
                high=148.0,
                low=149.0,
                close=149.0,
                volume=100,
            )

    def test_validation_negative_volume(self) -> None:
        """OHLCVBar rejects negative volume."""
        with pytest.raises(ValueError, match="volume.*must be >= 0"):
            OHLCVBar(
                ticker="AAPL",
                date=datetime.date(2025, 1, 15),
                open=150.0,
                high=155.0,
                low=149.0,
                close=153.0,
                volume=-1,
            )

    def test_bar_is_frozen(self) -> None:
        """OHLCVBar is immutable (frozen dataclass)."""
        bar = OHLCVBar(
            ticker="AAPL",
            date=datetime.date(2025, 1, 15),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1_000_000,
        )
        with pytest.raises(AttributeError):
            bar.close = 999.0  # type: ignore[misc]
