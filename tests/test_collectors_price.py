"""Tests for Yahoo Finance daily OHLCV bar downloader."""

from __future__ import annotations

import datetime
from unittest.mock import patch

import pandas as pd

from smaps.collectors.price import fetch_daily_bars
from smaps.models import OHLCVBar


class TestFetchDailyBars:
    """Tests for fetch_daily_bars with mocked yfinance."""

    def _make_mock_df(self) -> pd.DataFrame:
        """Create a mock DataFrame mimicking yfinance output."""
        index = pd.DatetimeIndex(
            [
                pd.Timestamp("2025-01-13"),
                pd.Timestamp("2025-01-14"),
                pd.Timestamp("2025-01-15"),
            ]
        )
        return pd.DataFrame(
            {
                "Open": [150.0, 152.0, 153.0],
                "High": [155.0, 156.0, 157.0],
                "Low": [149.0, 151.0, 152.0],
                "Close": [153.0, 154.0, 155.0],
                "Volume": [1000000, 1200000, 1100000],
            },
            index=index,
        )

    @patch("smaps.collectors.price.yf.download")
    def test_returns_ohlcv_bars(self, mock_download: object) -> None:
        """fetch_daily_bars returns a list of OHLCVBar objects."""
        mock_download.return_value = self._make_mock_df()  # type: ignore[union-attr]

        bars = fetch_daily_bars(
            "AAPL",
            datetime.date(2025, 1, 13),
            datetime.date(2025, 1, 15),
        )

        assert len(bars) == 3
        assert all(isinstance(b, OHLCVBar) for b in bars)

    @patch("smaps.collectors.price.yf.download")
    def test_parses_fields_correctly(self, mock_download: object) -> None:
        """fetch_daily_bars correctly maps DataFrame columns to OHLCVBar fields."""
        mock_download.return_value = self._make_mock_df()  # type: ignore[union-attr]

        bars = fetch_daily_bars(
            "AAPL",
            datetime.date(2025, 1, 13),
            datetime.date(2025, 1, 15),
        )

        first = bars[0]
        assert first.ticker == "AAPL"
        assert first.date == datetime.date(2025, 1, 13)
        assert first.open == 150.0
        assert first.high == 155.0
        assert first.low == 149.0
        assert first.close == 153.0
        assert first.volume == 1000000

    @patch("smaps.collectors.price.yf.download")
    def test_empty_response(self, mock_download: object) -> None:
        """fetch_daily_bars returns empty list when yfinance returns no data."""
        mock_download.return_value = pd.DataFrame()  # type: ignore[union-attr]

        bars = fetch_daily_bars(
            "INVALID",
            datetime.date(2025, 1, 13),
            datetime.date(2025, 1, 15),
        )

        assert bars == []

    @patch("smaps.collectors.price.yf.download")
    def test_passes_auto_adjust(self, mock_download: object) -> None:
        """fetch_daily_bars calls yfinance with auto_adjust=True."""
        mock_download.return_value = self._make_mock_df()  # type: ignore[union-attr]

        fetch_daily_bars(
            "AAPL",
            datetime.date(2025, 1, 13),
            datetime.date(2025, 1, 15),
        )

        mock_download.assert_called_once()  # type: ignore[union-attr]
        call_kwargs = mock_download.call_args.kwargs  # type: ignore[union-attr]
        assert call_kwargs["auto_adjust"] is True

    @patch("smaps.collectors.price.yf.download")
    def test_ticker_propagated(self, mock_download: object) -> None:
        """fetch_daily_bars sets the ticker field on all returned bars."""
        mock_download.return_value = self._make_mock_df()  # type: ignore[union-attr]

        bars = fetch_daily_bars(
            "MSFT",
            datetime.date(2025, 1, 13),
            datetime.date(2025, 1, 15),
        )

        assert all(b.ticker == "MSFT" for b in bars)
