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


def migration_005_model_registry(conn: sqlite3.Connection) -> None:
    """Create the model_registry table."""
    conn.execute(
        """\
        CREATE TABLE IF NOT EXISTS model_registry (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT    NOT NULL,
            version         INTEGER NOT NULL,
            trained_at      TEXT    NOT NULL,  -- ISO datetime
            metrics_json    TEXT    NOT NULL,
            artifact_path   TEXT    NOT NULL
        )
        """
    )


def migration_006_predictions(conn: sqlite3.Connection) -> None:
    """Create the predictions table."""
    conn.execute(
        """\
        CREATE TABLE IF NOT EXISTS predictions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker              TEXT    NOT NULL,
            prediction_date     TEXT    NOT NULL,  -- YYYY-MM-DD
            direction           TEXT    NOT NULL,  -- UP or DOWN
            confidence          REAL    NOT NULL,
            model_version       TEXT    NOT NULL,
            feature_snapshot_id INTEGER,
            created_at          TEXT    NOT NULL   -- ISO datetime
        )
        """
    )


def migration_007_evaluations(conn: sqlite3.Connection) -> None:
    """Create the evaluations table."""
    conn.execute(
        """\
        CREATE TABLE IF NOT EXISTS evaluations (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id     INTEGER NOT NULL,
            actual_direction  TEXT    NOT NULL,  -- UP or DOWN
            is_correct        INTEGER NOT NULL,  -- 0 or 1
            evaluated_at      TEXT    NOT NULL   -- ISO datetime
        )
        """
    )


# Ordered list of all migrations. Index 0 = migration 1, etc.
MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, migration_001_initial),
    (2, migration_002_sentiment),
    (3, migration_003_fundamentals),
    (4, migration_004_feature_snapshots),
    (5, migration_005_model_registry),
    (6, migration_006_predictions),
    (7, migration_007_evaluations),
]
