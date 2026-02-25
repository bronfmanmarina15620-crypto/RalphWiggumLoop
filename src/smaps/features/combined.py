"""Unified feature vector combining all feature pipelines."""

from __future__ import annotations

import datetime
import sqlite3

from smaps.features.fundamental import FundamentalFeatures
from smaps.features.sentiment import SentimentFeatures
from smaps.features.technical import TechnicalFeatures

# Canonical set of feature keys, always returned regardless of data availability.
FEATURE_KEYS = frozenset(
    [
        # Technical (7)
        "return_1d",
        "return_5d",
        "return_10d",
        "ma_ratio_5_20",
        "volume_change_1d",
        "volatility_20d",
        "rsi_14",
        # Sentiment (2)
        "latest_sentiment_score",
        "sentiment_ma_5d",
        # Fundamental (3)
        "pe_ratio",
        "eps",
        "market_cap",
    ]
)


def build_features(
    conn: sqlite3.Connection, ticker: str, as_of_date: datetime.date
) -> dict[str, float]:
    """Return the full feature vector for *ticker* as of *as_of_date*.

    Merges technical, sentiment, and fundamental features into a single
    dictionary.  The key set is **always** the same (``FEATURE_KEYS``)
    regardless of data availability — missing values are represented as
    ``float('nan')`` (technical/fundamental) or ``0.0`` (sentiment).
    """
    technical = TechnicalFeatures(conn).transform(ticker, as_of_date)
    sentiment = SentimentFeatures(conn).transform(ticker, as_of_date)
    fundamental = FundamentalFeatures(conn).transform(ticker, as_of_date)

    merged: dict[str, float] = {}
    merged.update(technical)
    merged.update(sentiment)
    merged.update(fundamental)

    return merged
