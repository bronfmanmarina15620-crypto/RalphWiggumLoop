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
