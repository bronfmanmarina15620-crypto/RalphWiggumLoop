"""SQLite database helpers."""

from __future__ import annotations

import sqlite3


def get_connection(db_path: str = ":memory:") -> sqlite3.Connection:
    """Open (or create) a SQLite database and return the connection."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create core tables if they do not already exist (idempotent)."""
    conn.execute(
        """\
        CREATE TABLE IF NOT EXISTS ohlcv_daily (
            ticker    TEXT    NOT NULL,
            date      TEXT    NOT NULL,  -- YYYY-MM-DD
            open      REAL,
            high      REAL,
            low       REAL,
            close     REAL,
            adj_close REAL,
            volume    REAL,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    conn.commit()
