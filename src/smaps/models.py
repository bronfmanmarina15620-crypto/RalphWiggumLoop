"""Canonical data models for SMAPS."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import Enum


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


class Direction(Enum):
    """Predicted price direction."""

    UP = "UP"
    DOWN = "DOWN"


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """A single prediction for a ticker on a given date."""

    ticker: str
    prediction_date: datetime.date
    direction: Direction
    confidence: float  # 0.0 to 1.0
    model_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.direction, Direction):
            raise ValueError(
                f"direction must be a Direction enum, got {type(self.direction).__name__}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence ({self.confidence}) must be between 0.0 and 1.0"
            )


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Result of evaluating a prediction against the actual outcome."""

    prediction_id: int
    actual_direction: Direction
    is_correct: bool
    evaluated_at: datetime.datetime


@dataclass(frozen=True, slots=True)
class MetricsReport:
    """Rolling accuracy metrics for a ticker over a date window."""

    ticker: str
    window_start: datetime.date
    window_end: datetime.date
    accuracy: float
    precision_up: float
    recall_up: float
    precision_down: float
    recall_down: float
    total_predictions: int
    evaluated_predictions: int

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict."""
        return {
            "ticker": self.ticker,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "accuracy": self.accuracy,
            "precision_up": self.precision_up,
            "recall_up": self.recall_up,
            "precision_down": self.precision_down,
            "recall_down": self.recall_down,
            "total_predictions": self.total_predictions,
            "evaluated_predictions": self.evaluated_predictions,
        }
