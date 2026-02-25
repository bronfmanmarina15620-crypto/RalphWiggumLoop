"""Fundamental-derived features computed from fundamentals_daily data."""

from __future__ import annotations

import datetime
import sqlite3


class FundamentalFeatures:
    """Compute fundamental-derived features from stored fundamentals data.

    Implements the ``FeaturePipeline`` protocol.  Only uses rows dated
    ``<= as_of_date`` to prevent look-ahead bias.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def transform(
        self, ticker: str, as_of_date: datetime.date
    ) -> dict[str, float]:
        """Return fundamental features for *ticker* as of *as_of_date*."""
        row = self._load_latest(ticker, as_of_date)

        if row is None:
            return {
                "pe_ratio": float("nan"),
                "eps": float("nan"),
                "market_cap": float("nan"),
            }

        pe_ratio, eps, market_cap = row
        return {
            "pe_ratio": float(pe_ratio) if pe_ratio is not None else float("nan"),
            "eps": float(eps) if eps is not None else float("nan"),
            "market_cap": float(market_cap) if market_cap is not None else float("nan"),
        }

    def _load_latest(
        self, ticker: str, as_of_date: datetime.date
    ) -> tuple[float | None, float | None, float | None] | None:
        """Load the most recent fundamentals row up to *as_of_date*."""
        cur = self._conn.execute(
            "SELECT pe_ratio, eps, market_cap FROM fundamentals_daily "
            "WHERE ticker = ? AND date <= ? ORDER BY date DESC LIMIT 1",
            (ticker, as_of_date.isoformat()),
        )
        return cur.fetchone()  # type: ignore[return-value]
