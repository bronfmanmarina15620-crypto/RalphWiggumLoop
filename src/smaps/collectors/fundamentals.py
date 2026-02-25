"""Fundamentals data collector using yfinance."""

from __future__ import annotations

import datetime

import yfinance as yf

from smaps.models import Fundamentals


def fetch_fundamentals(ticker: str) -> Fundamentals:
    """Fetch key fundamental metrics for a ticker.

    Uses yfinance Ticker.info to retrieve PE ratio, market cap,
    EPS, and revenue. All numeric fields are optional and will be
    None if not available from the data source.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").

    Returns:
        Fundamentals dataclass with available metrics.
    """
    info: dict[str, object] = yf.Ticker(ticker).info  # type: ignore[assignment]

    def _float_or_none(key: str) -> float | None:
        val = info.get(key)
        if val is None:
            return None
        try:
            return float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    return Fundamentals(
        ticker=ticker,
        date=datetime.date.today(),
        pe_ratio=_float_or_none("trailingPE"),
        market_cap=_float_or_none("marketCap"),
        eps=_float_or_none("trailingEps"),
        revenue=_float_or_none("totalRevenue"),
    )
