"""SQLite database helpers with lightweight schema versioning."""

from __future__ import annotations

import sqlite3

from smaps.migrations import MIGRATIONS
from smaps.models import OHLCVBar, SentimentScore

SCHEMA_VERSION = 2


def get_connection(db_path: str = ":memory:") -> sqlite3.Connection:
    """Open (or create) a SQLite database and return the connection."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Return the current schema version, or 0 if the tracking table is missing."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    )
    if cur.fetchone() is None:
        return 0
    cur = conn.execute("SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1")
    row = cur.fetchone()
    return row[0] if row else 0


def set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Record the current schema version."""
    conn.execute("DELETE FROM schema_migrations")
    conn.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))
    conn.commit()


def migrate(conn: sqlite3.Connection) -> None:
    """Apply all pending migrations up to SCHEMA_VERSION."""
    current = get_schema_version(conn)
    for version, fn in MIGRATIONS:
        if version > current:
            fn(conn)
            conn.commit()


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tracking table, run pending migrations, and set version (idempotent)."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER NOT NULL)"
    )
    conn.commit()
    current = get_schema_version(conn)
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version ({current}) is newer than "
            f"code schema version ({SCHEMA_VERSION}). "
            f"Upgrade the application or use a compatible database."
        )
    migrate(conn)
    set_schema_version(conn, SCHEMA_VERSION)


def upsert_bars(conn: sqlite3.Connection, bars: list[OHLCVBar]) -> int:
    """Persist OHLCV bars with INSERT OR REPLACE (idempotent upsert).

    Returns the number of rows upserted.
    """
    conn.executemany(
        """\
        INSERT OR REPLACE INTO ohlcv_daily (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (b.ticker, b.date.isoformat(), b.open, b.high, b.low, b.close, b.volume)
            for b in bars
        ],
    )
    conn.commit()
    return len(bars)


def upsert_sentiment(conn: sqlite3.Connection, scores: list[SentimentScore]) -> int:
    """Persist sentiment scores with INSERT OR REPLACE (idempotent upsert).

    Returns the number of rows upserted.
    """
    conn.executemany(
        """\
        INSERT OR REPLACE INTO sentiment_daily (ticker, date, score, source)
        VALUES (?, ?, ?, ?)
        """,
        [
            (s.ticker, s.date.isoformat(), s.score, s.source)
            for s in scores
        ],
    )
    conn.commit()
    return len(scores)
