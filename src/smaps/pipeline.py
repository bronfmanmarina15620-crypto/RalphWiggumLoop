"""Daily pipeline orchestrator: ingest → predict → evaluate → retrain-if-needed."""

from __future__ import annotations

import argparse
import datetime
import logging
import sqlite3
import time
import uuid

from smaps.collectors.fundamentals import fetch_fundamentals
from smaps.collectors.price import fetch_daily_bars
from smaps.collectors.sentiment import fetch_sentiment
from smaps.db import (
    ensure_schema,
    get_connection,
    save_evaluation,
    save_prediction,
    upsert_bars,
    upsert_fundamentals,
    upsert_sentiment,
)
from smaps.evaluator import evaluate_prediction
from smaps.logging import get_logger
from smaps.model.predictor import predict
from smaps.model.registry import load_latest_model
from smaps.retrainer import retrain_with_validation, should_retrain


def run_pipeline(
    tickers: list[str],
    date: datetime.date,
    db_path: str = ":memory:",
    models_dir: str = "models",
    lookback_days: int = 180,
) -> dict[str, object]:
    """Run the full daily pipeline: ingest → predict → evaluate → retrain-if-needed.

    Each step is logged with timing and *run_id*.  Failure in one ticker
    does not block others.

    Args:
        tickers: List of ticker symbols to process.
        date: The pipeline date (predictions are for this date).
        db_path: SQLite database path.
        models_dir: Directory for model artifact storage.
        lookback_days: Calendar days to look back for data ingestion.

    Returns:
        A dict with *run_id*, *date*, per-ticker step results, and total
        elapsed time.
    """
    run_id = uuid.uuid4().hex[:8]
    logger = get_logger("smaps.pipeline", run_id=run_id)

    t_pipeline = time.monotonic()
    logger.info(
        "pipeline_start tickers=%s date=%s",
        ",".join(tickers),
        date.isoformat(),
    )

    conn = get_connection(db_path)
    ensure_schema(conn)

    start_date = date - datetime.timedelta(days=lookback_days)
    ticker_results: dict[str, dict[str, object]] = {}

    for ticker in tickers:
        logger.info("pipeline_ticker_start ticker=%s", ticker)
        result = _run_ticker(conn, ticker, date, start_date, models_dir, logger)
        ticker_results[ticker] = result

    elapsed = time.monotonic() - t_pipeline
    conn.close()

    logger.info(
        "pipeline_complete run_id=%s tickers=%d elapsed=%.2fs",
        run_id,
        len(tickers),
        elapsed,
    )

    return {
        "run_id": run_id,
        "date": date.isoformat(),
        "tickers": ticker_results,
        "elapsed": elapsed,
    }


def _run_ticker(
    conn: sqlite3.Connection,
    ticker: str,
    date: datetime.date,
    start_date: datetime.date,
    models_dir: str,
    logger: logging.Logger,
) -> dict[str, object]:
    """Run all pipeline steps for a single ticker.

    Each step is wrapped in try/except so that a failing step does not
    prevent subsequent steps from executing.
    """
    result: dict[str, object] = {}

    # Step 1: Ingest
    t0 = time.monotonic()
    try:
        bars = fetch_daily_bars(ticker, start_date, date)
        upsert_bars(conn, bars)
        score = fetch_sentiment(ticker, date)
        upsert_sentiment(conn, [score])
        fund = fetch_fundamentals(ticker)
        upsert_fundamentals(conn, [fund])
        result["ingest"] = "ok"
        logger.info(
            "step=ingest ticker=%s status=ok elapsed=%.2fs",
            ticker,
            time.monotonic() - t0,
        )
    except Exception:
        result["ingest"] = "error"
        logger.exception("step=ingest ticker=%s status=error", ticker)

    # Step 1b: Auto-train if no model exists yet
    if load_latest_model(conn, ticker) is None:
        t0 = time.monotonic()
        try:
            retrain_with_validation(conn, ticker, models_dir)
            logger.info(
                "step=auto_train ticker=%s status=ok elapsed=%.2fs",
                ticker,
                time.monotonic() - t0,
            )
        except Exception:
            logger.exception("step=auto_train ticker=%s status=error", ticker)

    # Step 2: Predict (features built internally by predict())
    t0 = time.monotonic()
    try:
        prediction = predict(conn, ticker, date)
        pred_id = save_prediction(conn, prediction)
        result["predict"] = "ok"
        result["prediction_id"] = pred_id
        result["direction"] = prediction.direction.value
        result["confidence"] = prediction.confidence
        logger.info(
            "step=predict ticker=%s status=ok direction=%s confidence=%.4f "
            "elapsed=%.2fs",
            ticker,
            prediction.direction.value,
            prediction.confidence,
            time.monotonic() - t0,
        )
    except Exception:
        result["predict"] = "error"
        logger.exception("step=predict ticker=%s status=error", ticker)

    # Step 3: Evaluate past unevaluated predictions
    t0 = time.monotonic()
    try:
        evaluated_count = _evaluate_pending(conn, ticker)
        result["evaluate"] = "ok"
        result["evaluated_count"] = evaluated_count
        logger.info(
            "step=evaluate ticker=%s status=ok evaluated=%d elapsed=%.2fs",
            ticker,
            evaluated_count,
            time.monotonic() - t0,
        )
    except Exception:
        result["evaluate"] = "error"
        logger.exception("step=evaluate ticker=%s status=error", ticker)

    # Step 4: Retrain if needed
    t0 = time.monotonic()
    try:
        if should_retrain(conn, ticker, as_of_date=date):
            record = retrain_with_validation(conn, ticker, models_dir=models_dir)
            retrain_status = "retrained" if record else "rollback"
        else:
            retrain_status = "not_needed"
        result["retrain"] = retrain_status
        logger.info(
            "step=retrain ticker=%s status=%s elapsed=%.2fs",
            ticker,
            retrain_status,
            time.monotonic() - t0,
        )
    except Exception:
        result["retrain"] = "error"
        logger.exception("step=retrain ticker=%s status=error", ticker)

    return result


def _evaluate_pending(conn: sqlite3.Connection, ticker: str) -> int:
    """Evaluate all pending (unevaluated) predictions for *ticker*.

    Returns the number of successfully evaluated predictions.
    Predictions that cannot be evaluated yet (missing price data)
    are silently skipped.
    """
    cur = conn.execute(
        "SELECT p.id FROM predictions p "
        "LEFT JOIN evaluations e ON e.prediction_id = p.id "
        "WHERE p.ticker = ? AND e.id IS NULL",
        (ticker,),
    )
    pending_ids = [row[0] for row in cur.fetchall()]

    evaluated = 0
    for pred_id in pending_ids:
        try:
            eval_result = evaluate_prediction(conn, pred_id)
            save_evaluation(conn, eval_result)
            evaluated += 1
        except ValueError:
            pass  # Missing price data — can't evaluate yet

    return evaluated


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="smaps.pipeline",
        description="Run the SMAPS daily pipeline: ingest → predict → evaluate → retrain.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated ticker symbols (e.g. AAPL,MSFT). "
        "Defaults to SMAPS_TICKERS env var or config default.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Pipeline date in YYYY-MM-DD format (e.g. 2025-01-15). "
        "Defaults to today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log pipeline steps without executing them.",
    )
    return parser


def _run_dry(tickers: list[str], date: datetime.date) -> None:
    """Log pipeline steps without executing them."""
    logger = get_logger("smaps.pipeline", run_id="dry-run")
    logger.info(
        "dry_run tickers=%s date=%s",
        ",".join(tickers),
        date.isoformat(),
    )
    for ticker in tickers:
        for step in ("ingest", "predict", "evaluate", "retrain"):
            logger.info("dry_run step=%s ticker=%s — skipped", step, ticker)
    logger.info("dry_run complete — no changes made")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``python -m smaps.pipeline``."""
    from smaps.config import Settings

    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = Settings()

    tickers = (
        [t.strip() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else settings.tickers
    )

    if args.date:
        pipeline_date = datetime.date.fromisoformat(args.date)
    else:
        pipeline_date = datetime.date.today()

    if args.dry_run:
        _run_dry(tickers, pipeline_date)
        return

    result = run_pipeline(
        tickers=tickers,
        date=pipeline_date,
        db_path=settings.db_path,
    )

    try:
        from smaps.notifier import send_telegram_update, send_twitter_update
        send_twitter_update(result, settings)
        send_telegram_update(result, settings)
    except Exception:
        logging.getLogger(__name__).exception("notification_failed")


if __name__ == "__main__":
    main()
