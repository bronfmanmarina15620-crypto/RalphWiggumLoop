"""Model artifact persistence and versioned registry."""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
from pathlib import Path
from typing import NamedTuple

import joblib  # type: ignore[import-untyped]

from smaps.model.trainer import TrainedModel


class ModelRecord(NamedTuple):
    """A row from the model_registry table."""

    id: int | None
    ticker: str
    version: int
    trained_at: datetime.datetime
    metrics_json: str
    artifact_path: str


def _next_version(conn: sqlite3.Connection, ticker: str) -> int:
    """Return the next model version number for a ticker."""
    cur = conn.execute(
        "SELECT MAX(version) FROM model_registry WHERE ticker = ?",
        (ticker,),
    )
    row = cur.fetchone()
    current = row[0] if row[0] is not None else 0
    return current + 1


def save_model(
    conn: sqlite3.Connection,
    ticker: str,
    model: TrainedModel,
    models_dir: str | Path = "models",
) -> ModelRecord:
    """Save a trained model artifact and register it in the database.

    The model is saved to ``models/<ticker>_v<N>.joblib`` where N is the
    auto-incremented version number.

    Returns:
        The ModelRecord for the newly saved model.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    version = _next_version(conn, ticker)
    filename = f"{ticker}_v{version}.joblib"
    artifact_path = str(models_path / filename)
    trained_at = datetime.datetime.now(tz=datetime.timezone.utc)
    metrics_json = json.dumps(model.metrics)

    # Save the TrainedModel to disk
    joblib.dump(model, artifact_path)

    cur = conn.execute(
        """\
        INSERT INTO model_registry (ticker, version, trained_at, metrics_json, artifact_path)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ticker, version, trained_at.isoformat(), metrics_json, artifact_path),
    )
    conn.commit()

    return ModelRecord(
        id=cur.lastrowid,
        ticker=ticker,
        version=version,
        trained_at=trained_at,
        metrics_json=metrics_json,
        artifact_path=artifact_path,
    )


def load_latest_model(
    conn: sqlite3.Connection,
    ticker: str,
) -> tuple[TrainedModel, ModelRecord] | None:
    """Load the most recent model for a ticker.

    Returns:
        Tuple of (TrainedModel, ModelRecord) or None if no model exists.
    """
    cur = conn.execute(
        "SELECT id, ticker, version, trained_at, metrics_json, artifact_path "
        "FROM model_registry WHERE ticker = ? ORDER BY version DESC LIMIT 1",
        (ticker,),
    )
    row = cur.fetchone()
    if row is None:
        return None

    record = ModelRecord(
        id=row[0],
        ticker=row[1],
        version=row[2],
        trained_at=datetime.datetime.fromisoformat(row[3]),
        metrics_json=row[4],
        artifact_path=row[5],
    )

    if not os.path.exists(record.artifact_path):
        return None

    model: TrainedModel = joblib.load(record.artifact_path)
    return model, record
