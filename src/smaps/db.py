"""SQLite database helpers with lightweight schema versioning."""

from __future__ import annotations

import datetime
import json
import sqlite3
from typing import NamedTuple

from smaps.migrations import MIGRATIONS
from pathlib import Path

from smaps.models import Direction, EvalResult, Fundamentals, OHLCVBar, PredictionResult, SentimentScore

SCHEMA_VERSION = 7


class FeatureSnapshot(NamedTuple):
    """A persisted feature vector snapshot."""

    id: int | None
    ticker: str
    feature_date: datetime.date
    features: dict[str, float]
    pipeline_version: str


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


def upsert_fundamentals(conn: sqlite3.Connection, rows: list[Fundamentals]) -> int:
    """Persist fundamentals with INSERT OR REPLACE (idempotent upsert).

    Returns the number of rows upserted.
    """
    conn.executemany(
        """\
        INSERT OR REPLACE INTO fundamentals_daily
            (ticker, date, pe_ratio, market_cap, eps, revenue)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (f.ticker, f.date.isoformat(), f.pe_ratio, f.market_cap, f.eps, f.revenue)
            for f in rows
        ],
    )
    conn.commit()
    return len(rows)


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


def save_feature_snapshot(
    conn: sqlite3.Connection,
    ticker: str,
    feature_date: datetime.date,
    features: dict[str, float],
    pipeline_version: str,
) -> int:
    """Persist a feature vector snapshot. Returns the row id."""
    cur = conn.execute(
        """\
        INSERT INTO feature_snapshots (ticker, feature_date, features_json, pipeline_version)
        VALUES (?, ?, ?, ?)
        """,
        (ticker, feature_date.isoformat(), json.dumps(features), pipeline_version),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def load_feature_snapshot(conn: sqlite3.Connection, snapshot_id: int) -> FeatureSnapshot | None:
    """Load a feature snapshot by id. Returns None if not found."""
    cur = conn.execute(
        "SELECT id, ticker, feature_date, features_json, pipeline_version "
        "FROM feature_snapshots WHERE id = ?",
        (snapshot_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return FeatureSnapshot(
        id=row[0],
        ticker=row[1],
        feature_date=datetime.date.fromisoformat(row[2]),
        features=json.loads(row[3]),
        pipeline_version=row[4],
    )


class PredictionRecord(NamedTuple):
    """A persisted prediction row."""

    id: int | None
    ticker: str
    prediction_date: datetime.date
    direction: Direction
    confidence: float
    model_version: str
    feature_snapshot_id: int | None
    created_at: str


def save_prediction(
    conn: sqlite3.Connection,
    prediction: PredictionResult,
    feature_snapshot_id: int | None = None,
) -> int:
    """Persist a prediction to the database. Returns the row id."""
    created_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    cur = conn.execute(
        """\
        INSERT INTO predictions
            (ticker, prediction_date, direction, confidence,
             model_version, feature_snapshot_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prediction.ticker,
            prediction.prediction_date.isoformat(),
            prediction.direction.value,
            prediction.confidence,
            prediction.model_version,
            feature_snapshot_id,
            created_at,
        ),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def load_prediction(conn: sqlite3.Connection, prediction_id: int) -> PredictionRecord | None:
    """Load a prediction by id. Returns None if not found."""
    cur = conn.execute(
        "SELECT id, ticker, prediction_date, direction, confidence, "
        "model_version, feature_snapshot_id, created_at "
        "FROM predictions WHERE id = ?",
        (prediction_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return PredictionRecord(
        id=row[0],
        ticker=row[1],
        prediction_date=datetime.date.fromisoformat(row[2]),
        direction=Direction(row[3]),
        confidence=row[4],
        model_version=row[5],
        feature_snapshot_id=row[6],
        created_at=row[7],
    )


class EvalRecord(NamedTuple):
    """A persisted evaluation row."""

    id: int | None
    prediction_id: int
    actual_direction: Direction
    is_correct: bool
    evaluated_at: str


def save_evaluation(conn: sqlite3.Connection, result: EvalResult) -> int:
    """Persist an evaluation result to the database. Returns the row id."""
    cur = conn.execute(
        """\
        INSERT INTO evaluations
            (prediction_id, actual_direction, is_correct, evaluated_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            result.prediction_id,
            result.actual_direction.value,
            int(result.is_correct),
            result.evaluated_at.isoformat(),
        ),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def load_evaluation(conn: sqlite3.Connection, eval_id: int) -> EvalRecord | None:
    """Load an evaluation by id. Returns None if not found."""
    cur = conn.execute(
        "SELECT id, prediction_id, actual_direction, is_correct, evaluated_at "
        "FROM evaluations WHERE id = ?",
        (eval_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return EvalRecord(
        id=row[0],
        prediction_id=row[1],
        actual_direction=Direction(row[2]),
        is_correct=bool(row[3]),
        evaluated_at=row[4],
    )


def get_latest_predictions(
    conn: sqlite3.Connection,
    ticker: str | None = None,
) -> list[PredictionRecord]:
    """Return the most recent prediction per ticker.

    If *ticker* is given, return only the prediction for that ticker.
    Results are ordered by prediction_date descending.
    """
    if ticker:
        cur = conn.execute(
            "SELECT id, ticker, prediction_date, direction, confidence, "
            "model_version, feature_snapshot_id, created_at "
            "FROM predictions WHERE ticker = ? "
            "ORDER BY prediction_date DESC, id DESC LIMIT 1",
            (ticker,),
        )
    else:
        # Latest prediction per ticker via subquery
        cur = conn.execute(
            "SELECT p.id, p.ticker, p.prediction_date, p.direction, "
            "p.confidence, p.model_version, p.feature_snapshot_id, p.created_at "
            "FROM predictions p "
            "INNER JOIN ("
            "  SELECT ticker, MAX(id) AS max_id "
            "  FROM predictions GROUP BY ticker"
            ") latest ON p.id = latest.max_id "
            "ORDER BY p.prediction_date DESC",
        )
    rows = cur.fetchall()
    return [
        PredictionRecord(
            id=r[0],
            ticker=r[1],
            prediction_date=datetime.date.fromisoformat(r[2]),
            direction=Direction(r[3]),
            confidence=r[4],
            model_version=r[5],
            feature_snapshot_id=r[6],
            created_at=r[7],
        )
        for r in rows
    ]


def save_metrics_report(report: dict[str, object], reports_dir: str = "reports") -> Path:
    """Save a metrics report as a JSON file in the reports directory.

    The filename is ``metrics_<ticker>_<window_end>.json``.
    Returns the path to the written file.
    """
    path = Path(reports_dir)
    path.mkdir(parents=True, exist_ok=True)
    ticker = report.get("ticker", "unknown")
    window_end = report.get("window_end", "unknown")
    filename = f"metrics_{ticker}_{window_end}.json"
    filepath = path / filename
    filepath.write_text(json.dumps(report, indent=2))
    return filepath
