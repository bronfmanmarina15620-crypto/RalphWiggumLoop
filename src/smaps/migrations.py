"""Sequential schema migrations for SQLite."""

from __future__ import annotations

import sqlite3
from typing import Callable


def migration_001_initial(conn: sqlite3.Connection) -> None:
    """Create the ohlcv_daily table."""
    conn.execute(
        """\
        CREATE TABLE IF NOT EXISTS ohlcv_daily (
            ticker    TEXT    NOT NULL,
            date      TEXT    NOT NULL,  -- YYYY-MM-DD
            open      REAL,
            high      REAL,
            low       REAL,
            close     REAL,
            volume    INTEGER,
            PRIMARY KEY (ticker, date)
        )
        """
    )


def migration_002_sentiment(conn: sqlite3.Connection) -> None:
    """Create the sentiment_daily table."""
    conn.execute(
        """\
        CREATE TABLE IF NOT EXISTS sentiment_daily (
            ticker    TEXT    NOT NULL,
            date      TEXT    NOT NULL,  -- YYYY-MM-DD
            score     REAL    NOT NULL,
            source    TEXT    NOT NULL,
            PRIMARY KEY (ticker, date, source)
        )
        """
    )


def migration_003_fundamentals(conn: sqlite3.Connection) -> None:
    """Create the fundamentals_daily table."""
    conn.execute(
        """\
        CREATE TABLE IF NOT EXISTS fundamentals_daily (
            ticker      TEXT    NOT NULL,
            date        TEXT    NOT NULL,  -- YYYY-MM-DD
            pe_ratio    REAL,
            market_cap  REAL,
            eps         REAL,
            revenue     REAL,
            PRIMARY KEY (ticker, date)
        )
        """
    )


def migration_004_feature_snapshots(conn: sqlite3.Connection) -> None:
    """Create the feature_snapshots table."""
    conn.execute(
        """\
        CREATE TABLE IF NOT EXISTS feature_snapshots (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker            TEXT    NOT NULL,
            feature_date      TEXT    NOT NULL,  -- YYYY-MM-DD
            features_json     TEXT    NOT NULL,
            pipeline_version  TEXT    NOT NULL
        )
        """
    )


# Ordered list of all migrations. Index 0 = migration 1, etc.
MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, migration_001_initial),
    (2, migration_002_sentiment),
    (3, migration_003_fundamentals),
    (4, migration_004_feature_snapshots),
]
