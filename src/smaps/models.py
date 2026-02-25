"""Canonical data models for SMAPS."""

from __future__ import annotations

import datetime
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OHLCVBar:
    """A single daily OHLCV price bar."""

    ticker: str
    date: datetime.date
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(
                f"high ({self.high}) must be >= low ({self.low})"
            )
        if self.volume < 0:
            raise ValueError(
                f"volume ({self.volume}) must be >= 0"
            )


@dataclass(frozen=True, slots=True)
class SentimentScore:
    """A daily sentiment score for a ticker."""

    ticker: str
    date: datetime.date
    score: float  # -1.0 (bearish) to 1.0 (bullish)
    source: str

    def __post_init__(self) -> None:
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(
                f"score ({self.score}) must be between -1.0 and 1.0"
            )


@dataclass(frozen=True, slots=True)
class Fundamentals:
    """Key fundamental metrics for a ticker."""

    ticker: str
    date: datetime.date
    pe_ratio: float | None = None
    market_cap: float | None = None
    eps: float | None = None
    revenue: float | None = None
