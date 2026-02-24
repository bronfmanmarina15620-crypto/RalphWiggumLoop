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


# Ordered list of all migrations. Index 0 = migration 1, etc.
MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, migration_001_initial),
    (2, migration_002_sentiment),
]
