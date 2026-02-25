"""Multi-ticker ingestion orchestrator."""

from __future__ import annotations

import datetime
import logging
import sqlite3
import time

from smaps.collectors.fundamentals import fetch_fundamentals
from smaps.collectors.price import fetch_daily_bars
from smaps.collectors.sentiment import fetch_sentiment
from smaps.db import (
    get_connection,
    ensure_schema,
    upsert_bars,
    upsert_fundamentals,
    upsert_sentiment,
)
from smaps.logging import get_logger


def ingest_all(
    tickers: list[str],
    start: datetime.date,
    end: datetime.date,
    db_path: str = ":memory:",
) -> dict[str, list[str]]:
    """Ingest price, sentiment, and fundamentals for all tickers.

    Error in one ticker does not block others. Each step is logged
    with timing information.

    Args:
        tickers: List of ticker symbols to ingest.
        start: Start date for price data (inclusive).
        end: End date for price data (inclusive).
        db_path: SQLite database path.

    Returns:
        Dict with "succeeded" and "failed" ticker lists.
    """
    logger = get_logger("smaps.collectors.ingest")
    conn = get_connection(db_path)
    ensure_schema(conn)

    succeeded: list[str] = []
    failed: list[str] = []

    for ticker in tickers:
        logger.info("Starting ingestion for %s", ticker)
        try:
            _ingest_ticker(ticker, start, end, conn, logger)
            succeeded.append(ticker)
            logger.info("Completed ingestion for %s", ticker)
        except Exception:
            failed.append(ticker)
            logger.exception("Failed ingestion for %s", ticker)

    conn.close()
    logger.info(
        "Ingestion complete: %d succeeded, %d failed",
        len(succeeded),
        len(failed),
    )
    return {"succeeded": succeeded, "failed": failed}


def _ingest_ticker(
    ticker: str,
    start: datetime.date,
    end: datetime.date,
    conn: sqlite3.Connection,
    logger: logging.Logger,
) -> None:
    """Ingest all data sources for a single ticker."""
    # Price bars
    t0 = time.monotonic()
    bars = fetch_daily_bars(ticker, start, end)
    upsert_bars(conn, bars)
    elapsed = time.monotonic() - t0
    logger.info("  price: %d bars in %.2fs", len(bars), elapsed)

    # Sentiment
    t0 = time.monotonic()
    score = fetch_sentiment(ticker, end)
    upsert_sentiment(conn, [score])
    elapsed = time.monotonic() - t0
    logger.info("  sentiment: score=%.4f in %.2fs", score.score, elapsed)

    # Fundamentals
    t0 = time.monotonic()
    fund = fetch_fundamentals(ticker)
    upsert_fundamentals(conn, [fund])
    elapsed = time.monotonic() - t0
    logger.info("  fundamentals: pe=%s in %.2fs", fund.pe_ratio, elapsed)
