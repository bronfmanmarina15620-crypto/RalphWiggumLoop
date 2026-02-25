"""Self-learning loop: retrain trigger and automated retraining."""

from __future__ import annotations

import datetime
import sqlite3
import time

import pandas as pd

from smaps.evaluator import compute_metrics
from smaps.features.combined import build_features
from smaps.logging import get_logger
from smaps.model.registry import ModelRecord, save_model
from smaps.model.trainer import TrainedModel, train_model

logger = get_logger(__name__)


def should_retrain(
    conn: sqlite3.Connection,
    ticker: str,
    threshold: float = 0.50,
    window_days: int = 30,
    as_of_date: datetime.date | None = None,
) -> bool:
    """Check whether the model for *ticker* should be retrained.

    Returns ``True`` when the rolling accuracy over the last *window_days*
    drops below *threshold*.  A structured log event is emitted when
    retraining is triggered.

    If there are no evaluated predictions in the window, returns ``False``
    (no evidence of degradation).
    """
    report = compute_metrics(
        conn, ticker, window_days=window_days, as_of_date=as_of_date,
    )

    if report.evaluated_predictions == 0:
        logger.info(
            "retrain_check ticker=%s result=skip reason=no_evaluated_predictions "
            "window_days=%d",
            ticker,
            window_days,
        )
        return False

    if report.accuracy < threshold:
        logger.warning(
            "retrain_triggered ticker=%s accuracy=%.4f threshold=%.4f "
            "evaluated=%d window_days=%d",
            ticker,
            report.accuracy,
            threshold,
            report.evaluated_predictions,
            window_days,
        )
        return True

    logger.info(
        "retrain_check ticker=%s result=ok accuracy=%.4f threshold=%.4f "
        "evaluated=%d window_days=%d",
        ticker,
        report.accuracy,
        threshold,
        report.evaluated_predictions,
        window_days,
    )
    return False


def _get_trading_dates(conn: sqlite3.Connection, ticker: str) -> list[datetime.date]:
    """Return all distinct dates with OHLCV data for *ticker*, sorted ascending."""
    cur = conn.execute(
        "SELECT DISTINCT date FROM ohlcv_daily WHERE ticker = ? ORDER BY date ASC",
        (ticker,),
    )
    return [datetime.date.fromisoformat(row[0]) for row in cur.fetchall()]


def retrain(
    conn: sqlite3.Connection,
    ticker: str,
    models_dir: str = "models",
) -> ModelRecord:
    """Retrain the model for *ticker* using all available historical data.

    Fetches all OHLCV dates for the ticker, builds features for each date,
    computes next-day direction labels, trains a new model, and saves it
    with an incremented version number.

    Args:
        conn: SQLite connection with schema already applied.
        ticker: Stock ticker symbol.
        models_dir: Directory for model artifact storage.

    Returns:
        The :class:`ModelRecord` for the newly trained and saved model.

    Raises:
        ValueError: If insufficient data is available (need at least 2
            trading days for features + labels).
    """
    t0 = time.monotonic()

    # 1. Get all trading dates for this ticker
    dates = _get_trading_dates(conn, ticker)
    if len(dates) < 2:
        raise ValueError(
            f"Insufficient data for ticker '{ticker}': need at least 2 "
            f"trading days, found {len(dates)}."
        )

    logger.info(
        "retrain_start ticker=%s trading_days=%d date_range=%s..%s",
        ticker,
        len(dates),
        dates[0].isoformat(),
        dates[-1].isoformat(),
    )

    # 2. Build a close-price lookup for label computation
    cur = conn.execute(
        "SELECT date, close FROM ohlcv_daily WHERE ticker = ? ORDER BY date ASC",
        (ticker,),
    )
    close_by_date = {row[0]: row[1] for row in cur.fetchall()}

    # 3. Build features and labels for each date that has a next-day price
    feature_rows: list[dict[str, float]] = []
    labels: list[int] = []

    for i, dt in enumerate(dates[:-1]):
        next_dt = dates[i + 1]
        close_today = close_by_date[dt.isoformat()]
        close_next = close_by_date[next_dt.isoformat()]
        label = 1 if close_next >= close_today else 0  # UP=1, DOWN=0

        features = build_features(conn, ticker, dt)
        feature_rows.append(features)
        labels.append(label)

    # 4. Train the model
    features_df = pd.DataFrame(feature_rows)
    labels_series = pd.Series(labels)
    model: TrainedModel = train_model(features_df, labels_series)

    # 5. Save with incremented version
    record = save_model(conn, ticker, model, models_dir=models_dir)

    elapsed = time.monotonic() - t0
    logger.info(
        "retrain_complete ticker=%s version=%d accuracy=%.4f "
        "train_samples=%d elapsed=%.2fs",
        ticker,
        record.version,
        model.metrics.get("accuracy", 0.0),
        len(feature_rows),
        elapsed,
    )

    return record
