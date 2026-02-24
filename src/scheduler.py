"""Background scheduler: orchestrates the daily prediction/validation/learning pipeline."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from .learner import ModelVersion, run_daily_learning
from .predictor import run_daily_prediction
from .validator import run_daily_validation

logger = logging.getLogger(__name__)

# Symbols to track — extend this list to monitor additional tickers.
TRACKED_SYMBOLS: list[str] = ["AAPL"]

# Wall-clock time (UTC) at which the daily workflow fires.
DAILY_RUN_HOUR: int = 22    # 10 PM UTC — ~2 h after US market close
DAILY_RUN_MINUTE: int = 0


def run_daily_workflow(symbol: str) -> None:
    """Run the full prediction → validation → learning pipeline for *symbol*.

    Each step is logged so that the outcome of every prediction and every
    learning event can be reviewed in the application log.

    Args:
        symbol: Ticker symbol (e.g. ``'AAPL'``).
    """
    logger.info("=== Daily workflow START: %s ===", symbol)

    # Step 1 — Predict
    try:
        prediction = run_daily_prediction(symbol)
        logger.info(
            "Prediction: %s %s | up=%.2f%% down=%.2f%%",
            symbol,
            prediction.label,
            prediction.up_probability * 100,
            prediction.down_probability * 100,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Prediction failed for %s: %s", symbol, exc)
        logger.info("=== Daily workflow ABORTED: %s ===", symbol)
        return

    # Step 2 — Validate
    try:
        result = run_daily_validation(prediction)
        logger.info(
            "Validation: predicted=%s actual=%s correct=%s"
            " (close %.2f → %.2f)",
            result.prediction.label,
            result.actual_label,
            result.correct,
            result.prediction_close,
            result.next_close,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Validation failed for %s: %s", symbol, exc)
        logger.info("=== Daily workflow ABORTED: %s ===", symbol)
        return

    # Step 3 — Learn
    try:
        model_version: ModelVersion | None = run_daily_learning(result)
        if model_version is not None:
            logger.info(
                "Model retrained: version=%s trained_at=%s samples=%d path=%s",
                model_version.version_id,
                model_version.trained_at,
                model_version.training_samples,
                model_version.model_path,
            )
        else:
            logger.info("Prediction was correct — no retraining needed for %s", symbol)
    except Exception as exc:  # noqa: BLE001
        logger.error("Learning failed for %s: %s", symbol, exc)

    logger.info("=== Daily workflow END: %s ===", symbol)


def _seconds_until_next_run() -> float:
    """Return seconds until the next scheduled run at ``DAILY_RUN_HOUR:DAILY_RUN_MINUTE`` UTC."""
    now = datetime.now(tz=timezone.utc)
    next_run = now.replace(
        hour=DAILY_RUN_HOUR,
        minute=DAILY_RUN_MINUTE,
        second=0,
        microsecond=0,
    )
    if next_run <= now:
        next_run += timedelta(days=1)
    return (next_run - now).total_seconds()


def run_scheduler(symbols: list[str] | None = None) -> None:
    """Start the daily scheduling loop.

    Blocks indefinitely, waking at ``DAILY_RUN_HOUR:DAILY_RUN_MINUTE`` UTC each
    day to run :func:`run_daily_workflow` for every tracked symbol.

    Alternatively, invoke :func:`run_daily_workflow` directly from a system
    cron job (e.g. ``0 22 * * 1-5 python -m src.scheduler --once``) to
    delegate scheduling to the OS.

    Args:
        symbols: List of ticker symbols to track.  Defaults to
            :data:`TRACKED_SYMBOLS` when ``None``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    active_symbols: list[str] = symbols if symbols is not None else TRACKED_SYMBOLS
    logger.info(
        "Scheduler started. Daily run at %02d:%02d UTC | symbols: %s",
        DAILY_RUN_HOUR,
        DAILY_RUN_MINUTE,
        active_symbols,
    )
    while True:
        wait = _seconds_until_next_run()
        logger.info("Next run in %.0f s (%.2f h)", wait, wait / 3600)
        time.sleep(wait)
        logger.info("Scheduler woke — running daily workflow")
        for symbol in active_symbols:
            run_daily_workflow(symbol)


if __name__ == "__main__":
    import sys

    # Support --once flag for cron-based invocation: run the workflow now and exit.
    if "--once" in sys.argv:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        for sym in TRACKED_SYMBOLS:
            run_daily_workflow(sym)
    else:
        run_scheduler()
