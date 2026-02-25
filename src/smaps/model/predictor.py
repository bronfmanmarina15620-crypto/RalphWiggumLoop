"""Daily prediction function: loads model, builds features, returns prediction."""

from __future__ import annotations

import datetime
import sqlite3

from smaps.features.combined import build_features
from smaps.model.registry import load_latest_model
from smaps.models import PredictionResult


def predict(
    conn: sqlite3.Connection,
    ticker: str,
    date: datetime.date,
) -> PredictionResult:
    """Predict next-day price direction for *ticker* as of *date*.

    Loads the latest trained model for the ticker, builds the feature
    vector, and returns a :class:`PredictionResult`.

    Args:
        conn: SQLite connection with schema already applied.
        ticker: Stock ticker symbol.
        date: The as-of date for feature computation and prediction.

    Returns:
        A PredictionResult with direction, confidence, and model version.

    Raises:
        RuntimeError: If no trained model exists for the ticker.
    """
    result = load_latest_model(conn, ticker)
    if result is None:
        raise RuntimeError(
            f"No trained model found for ticker '{ticker}'. "
            "Train a model first using train_model()."
        )

    model, record = result

    features = build_features(conn, ticker, date)
    direction, confidence = model.predict(features)

    return PredictionResult(
        ticker=ticker,
        prediction_date=date,
        direction=direction,
        confidence=confidence,
        model_version=f"v{record.version}",
    )
