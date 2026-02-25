"""Evaluate predictions against realized market outcomes."""

from __future__ import annotations

import datetime
import sqlite3

from smaps.db import load_prediction
from smaps.models import Direction, EvalResult, MetricsReport


def _find_next_trading_day_close(
    conn: sqlite3.Connection,
    ticker: str,
    after_date: datetime.date,
    max_lookahead_days: int = 10,
) -> tuple[float, datetime.date] | None:
    """Find the closing price on the next available trading day after *after_date*.

    Skips weekends and holidays by looking for the next row in ohlcv_daily
    with date > after_date.  Returns (close, trade_date) or None if no
    trading day is found within *max_lookahead_days*.
    """
    end_date = after_date + datetime.timedelta(days=max_lookahead_days)
    cur = conn.execute(
        "SELECT close, date FROM ohlcv_daily "
        "WHERE ticker = ? AND date > ? AND date <= ? "
        "ORDER BY date ASC LIMIT 1",
        (ticker, after_date.isoformat(), end_date.isoformat()),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return float(row[0]), datetime.date.fromisoformat(row[1])


def _get_close_on_date(
    conn: sqlite3.Connection,
    ticker: str,
    target_date: datetime.date,
) -> float | None:
    """Return the closing price on *target_date*, or None if no bar exists."""
    cur = conn.execute(
        "SELECT close FROM ohlcv_daily WHERE ticker = ? AND date = ?",
        (ticker, target_date.isoformat()),
    )
    row = cur.fetchone()
    return float(row[0]) if row else None


def evaluate_prediction(
    conn: sqlite3.Connection,
    prediction_id: int,
) -> EvalResult:
    """Compare a stored prediction to the actual price movement.

    Determines the actual direction by comparing the close on the
    prediction date to the close on the next trading day (skipping
    weekends and holidays).

    Args:
        conn: SQLite connection with schema already applied.
        prediction_id: The id of the prediction to evaluate.

    Returns:
        An EvalResult with actual_direction, is_correct, and evaluated_at.

    Raises:
        ValueError: If the prediction does not exist.
        ValueError: If price data is not available for evaluation.
    """
    prediction = load_prediction(conn, prediction_id)
    if prediction is None:
        raise ValueError(f"Prediction with id {prediction_id} not found")

    # Get the close price on the prediction date
    pred_close = _get_close_on_date(conn, prediction.ticker, prediction.prediction_date)
    if pred_close is None:
        raise ValueError(
            f"No price data on prediction date {prediction.prediction_date} "
            f"for {prediction.ticker}"
        )

    # Get the close price on the next trading day
    next_day = _find_next_trading_day_close(
        conn, prediction.ticker, prediction.prediction_date
    )
    if next_day is None:
        raise ValueError(
            f"No next-day price data after {prediction.prediction_date} "
            f"for {prediction.ticker}"
        )

    next_close, _ = next_day

    # Determine actual direction
    if next_close >= pred_close:
        actual_direction = Direction.UP
    else:
        actual_direction = Direction.DOWN

    is_correct = prediction.direction == actual_direction
    evaluated_at = datetime.datetime.now(tz=datetime.timezone.utc)

    return EvalResult(
        prediction_id=prediction_id,
        actual_direction=actual_direction,
        is_correct=is_correct,
        evaluated_at=evaluated_at,
    )


def compute_metrics(
    conn: sqlite3.Connection,
    ticker: str,
    window_days: int = 90,
    as_of_date: datetime.date | None = None,
) -> MetricsReport:
    """Compute rolling accuracy, precision, and recall over a date window.

    Args:
        conn: SQLite connection with schema already applied.
        ticker: The stock ticker to compute metrics for.
        window_days: Number of days to look back from *as_of_date*.
        as_of_date: End of the window (defaults to today).

    Returns:
        A MetricsReport with accuracy, precision, and recall per UP/DOWN class.
    """
    end_date = as_of_date or datetime.date.today()
    start_date = end_date - datetime.timedelta(days=window_days)

    # Get all prediction IDs and directions in the window
    cur = conn.execute(
        "SELECT id, direction FROM predictions "
        "WHERE ticker = ? AND prediction_date >= ? AND prediction_date <= ? "
        "ORDER BY prediction_date",
        (ticker, start_date.isoformat(), end_date.isoformat()),
    )
    predictions = [(row[0], Direction(row[1])) for row in cur.fetchall()]
    total_predictions = len(predictions)

    # Evaluate each prediction and build confusion matrix
    tp_up = 0
    fp_up = 0
    fn_up = 0
    tp_down = 0
    evaluated = 0

    for pred_id, pred_direction in predictions:
        try:
            result = evaluate_prediction(conn, pred_id)
        except ValueError:
            continue
        evaluated += 1

        if pred_direction == Direction.UP:
            if result.actual_direction == Direction.UP:
                tp_up += 1
            else:
                fp_up += 1
        else:
            if result.actual_direction == Direction.DOWN:
                tp_down += 1
            else:
                fn_up += 1

    # Compute metrics (0.0 for empty denominators)
    accuracy = (tp_up + tp_down) / evaluated if evaluated > 0 else 0.0
    precision_up = tp_up / (tp_up + fp_up) if (tp_up + fp_up) > 0 else 0.0
    recall_up = tp_up / (tp_up + fn_up) if (tp_up + fn_up) > 0 else 0.0
    # FP_down = FN_up (predicted DOWN, actual UP)
    precision_down = tp_down / (tp_down + fn_up) if (tp_down + fn_up) > 0 else 0.0
    # FN_down = FP_up (predicted UP, actual DOWN)
    recall_down = tp_down / (tp_down + fp_up) if (tp_down + fp_up) > 0 else 0.0

    return MetricsReport(
        ticker=ticker,
        window_start=start_date,
        window_end=end_date,
        accuracy=accuracy,
        precision_up=precision_up,
        recall_up=recall_up,
        precision_down=precision_down,
        recall_down=recall_down,
        total_predictions=total_predictions,
        evaluated_predictions=evaluated,
    )
