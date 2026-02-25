"""FastAPI application exposing SMAPS predictions and performance data."""

from __future__ import annotations

import sqlite3

from fastapi import FastAPI, Query

from smaps.config import Settings
from smaps.db import ensure_schema, get_latest_predictions

app = FastAPI(title="SMAPS API", version="0.1.0")


def _get_conn() -> sqlite3.Connection:
    """Return a DB connection using the configured db_path.

    Uses ``check_same_thread=False`` because FastAPI runs sync endpoints
    in a thread-pool.
    """
    settings = Settings()
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_schema(conn)
    return conn


@app.get("/predictions/latest")
def predictions_latest(
    ticker: str | None = Query(default=None, description="Filter by ticker symbol"),
) -> list[dict[str, object]]:
    """Return the latest prediction for each ticker.

    Optionally filter by ``?ticker=AAPL``.
    """
    conn = _get_conn()
    try:
        records = get_latest_predictions(conn, ticker=ticker)
        return [
            {
                "ticker": r.ticker,
                "prediction_date": r.prediction_date.isoformat(),
                "direction": r.direction.value,
                "confidence": r.confidence,
                "model_version": r.model_version,
            }
            for r in records
        ]
    finally:
        conn.close()
