"""Tests for fundamentals data collector."""

from __future__ import annotations

import datetime
from unittest.mock import patch, MagicMock

from smaps.collectors.fundamentals import fetch_fundamentals
from smaps.models import Fundamentals


class TestFetchFundamentals:
    """Tests for fetch_fundamentals with mocked yfinance."""

    _MOCK_INFO: dict[str, object] = {
        "trailingPE": 28.5,
        "marketCap": 2800000000000,
        "trailingEps": 6.42,
        "totalRevenue": 394328000000,
        "shortName": "Apple Inc.",
    }

    _MOCK_INFO_PARTIAL: dict[str, object] = {
        "trailingPE": 15.3,
        "shortName": "Test Corp.",
    }

    _MOCK_INFO_EMPTY: dict[str, object] = {
        "shortName": "Empty Corp.",
    }

    @patch("smaps.collectors.fundamentals.yf.Ticker")
    def test_returns_fundamentals(self, mock_ticker_cls: MagicMock) -> None:
        """fetch_fundamentals returns a Fundamentals object."""
        mock_ticker_cls.return_value.info = self._MOCK_INFO

        result = fetch_fundamentals("AAPL")

        assert isinstance(result, Fundamentals)

    @patch("smaps.collectors.fundamentals.yf.Ticker")
    def test_parses_all_fields(self, mock_ticker_cls: MagicMock) -> None:
        """fetch_fundamentals correctly maps info dict to Fundamentals fields."""
        mock_ticker_cls.return_value.info = self._MOCK_INFO

        result = fetch_fundamentals("AAPL")

        assert result.ticker == "AAPL"
        assert result.date == datetime.date.today()
        assert result.pe_ratio == 28.5
        assert result.market_cap == 2800000000000
        assert result.eps == 6.42
        assert result.revenue == 394328000000

    @patch("smaps.collectors.fundamentals.yf.Ticker")
    def test_partial_info_returns_none_for_missing(
        self, mock_ticker_cls: MagicMock
    ) -> None:
        """Missing keys in info dict produce None values."""
        mock_ticker_cls.return_value.info = self._MOCK_INFO_PARTIAL

        result = fetch_fundamentals("TEST")

        assert result.pe_ratio == 15.3
        assert result.market_cap is None
        assert result.eps is None
        assert result.revenue is None

    @patch("smaps.collectors.fundamentals.yf.Ticker")
    def test_empty_info_all_none(self, mock_ticker_cls: MagicMock) -> None:
        """Info dict with no fundamental keys returns all None fields."""
        mock_ticker_cls.return_value.info = self._MOCK_INFO_EMPTY

        result = fetch_fundamentals("EMPTY")

        assert result.ticker == "EMPTY"
        assert result.pe_ratio is None
        assert result.market_cap is None
        assert result.eps is None
        assert result.revenue is None

    @patch("smaps.collectors.fundamentals.yf.Ticker")
    def test_ticker_passed_to_yfinance(
        self, mock_ticker_cls: MagicMock
    ) -> None:
        """fetch_fundamentals passes the ticker to yf.Ticker."""
        mock_ticker_cls.return_value.info = self._MOCK_INFO

        fetch_fundamentals("MSFT")

        mock_ticker_cls.assert_called_once_with("MSFT")

    @patch("smaps.collectors.fundamentals.yf.Ticker")
    def test_non_numeric_value_returns_none(
        self, mock_ticker_cls: MagicMock
    ) -> None:
        """Non-numeric values in info dict produce None."""
        mock_ticker_cls.return_value.info = {
            "trailingPE": "N/A",
            "marketCap": None,
            "trailingEps": "not a number",
            "totalRevenue": 100000,
        }

        result = fetch_fundamentals("BAD")

        assert result.pe_ratio is None
        assert result.market_cap is None
        assert result.eps is None
        assert result.revenue == 100000
