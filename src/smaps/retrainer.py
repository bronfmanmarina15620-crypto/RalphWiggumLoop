"""Self-learning loop: retrain trigger and automated retraining."""

from __future__ import annotations

import datetime
import sqlite3

from smaps.evaluator import compute_metrics
from smaps.logging import get_logger

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
