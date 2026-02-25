"""Sentiment-derived features computed from sentiment_daily data."""

from __future__ import annotations

import datetime
import sqlite3


class SentimentFeatures:
    """Compute sentiment-derived features from stored sentiment scores.

    Implements the ``FeaturePipeline`` protocol.  Only uses scores dated
    ``<= as_of_date`` to prevent look-ahead bias.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def transform(
        self, ticker: str, as_of_date: datetime.date
    ) -> dict[str, float]:
        """Return sentiment features for *ticker* as of *as_of_date*."""
        scores = self._load_scores(ticker, as_of_date)

        return {
            "latest_sentiment_score": scores[-1] if scores else 0.0,
            "sentiment_ma_5d": self._rolling_avg(scores, 5),
        }

    def _load_scores(
        self, ticker: str, as_of_date: datetime.date
    ) -> list[float]:
        """Load sentiment scores up to *as_of_date* (inclusive), ordered by date."""
        cur = self._conn.execute(
            "SELECT score FROM sentiment_daily "
            "WHERE ticker = ? AND date <= ? ORDER BY date ASC",
            (ticker, as_of_date.isoformat()),
        )
        return [float(row[0]) for row in cur.fetchall()]

    @staticmethod
    def _rolling_avg(values: list[float], window: int) -> float:
        """Average of the last *window* values, or 0.0 if empty."""
        if not values:
            return 0.0
        tail = values[-window:]
        return sum(tail) / len(tail)
