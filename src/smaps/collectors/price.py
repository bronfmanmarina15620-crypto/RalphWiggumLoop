"""Yahoo Finance daily OHLCV bar downloader."""

from __future__ import annotations

import datetime

import yfinance as yf

from smaps.models import OHLCVBar


def fetch_daily_bars(
    ticker: str,
    start: datetime.date,
    end: datetime.date,
) -> list[OHLCVBar]:
    """Fetch daily OHLCV bars from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        start: Start date (inclusive).
        end: End date (inclusive).

    Returns:
        List of OHLCVBar objects sorted by date ascending.
    """
    data = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + datetime.timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
    )

    # yfinance >= 0.2.31 returns multi-level columns for single tickers;
    # flatten so we can index by simple column name.
    if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
        data.columns = data.columns.droplevel(1)

    bars: list[OHLCVBar] = []
    for ts, row in data.iterrows():
        bar = OHLCVBar(
            ticker=ticker,
            date=ts.date(),  # type: ignore[union-attr]
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=int(row["Volume"]),
        )
        bars.append(bar)

    return bars
