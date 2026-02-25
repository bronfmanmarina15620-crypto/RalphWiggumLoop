"""Self-learning loop: retrain trigger and automated retraining."""

from __future__ import annotations

import datetime
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp  # type: ignore[import-untyped]

from smaps.evaluator import compute_metrics
from smaps.features.combined import FEATURE_KEYS, build_features
from smaps.logging import get_logger
from smaps.model.registry import ModelRecord, load_latest_model, save_model
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


def validate_oos(
    new_model: TrainedModel,
    features_df: pd.DataFrame,
    labels: pd.Series,  # type: ignore[type-arg]
    current_model: TrainedModel | None = None,
    oos_days: int = 30,
) -> tuple[bool, dict[str, float]]:
    """Out-of-sample validation gate for model promotion.

    Holds out the last *oos_days* samples as the OOS validation set and
    evaluates the candidate model's accuracy on it.

    If a *current_model* is provided, the new model is only promoted when
    its OOS accuracy **strictly exceeds** the current model's OOS accuracy.
    If no current model exists, the new model is always promoted.

    Args:
        new_model: Newly trained model candidate.
        features_df: Full feature matrix (rows=samples, ordered
            chronologically).
        labels: Binary labels (1=UP, 0=DOWN) aligned with *features_df*.
        current_model: Currently deployed model, or ``None`` if this is
            the first model for the ticker.
        oos_days: Number of most-recent samples to hold out for OOS
            validation.

    Returns:
        Tuple of ``(should_promote, metrics)`` where *should_promote* is
        a boolean and *metrics* is a dict containing at least
        ``new_oos_accuracy`` and ``oos_size``.
    """
    n = len(features_df)

    if n <= oos_days:
        logger.info(
            "oos_gate result=promote reason=insufficient_data "
            "samples=%d oos_days=%d",
            n,
            oos_days,
        )
        return True, {"promoted": 1.0, "oos_size": float(n)}

    oos_features = features_df.iloc[-oos_days:]
    oos_labels = labels.iloc[-oos_days:]

    # Evaluate new model on OOS
    oos_filled = oos_features[new_model.feature_names].fillna(0.0)
    new_preds = new_model.pipeline.predict(oos_filled)
    new_accuracy = float(np.mean(new_preds == oos_labels.values))

    metrics: dict[str, float] = {
        "new_oos_accuracy": new_accuracy,
        "oos_size": float(len(oos_labels)),
    }

    if current_model is None:
        metrics["promoted"] = 1.0
        logger.info(
            "oos_gate result=promote reason=no_current_model "
            "new_oos_accuracy=%.4f oos_size=%d",
            new_accuracy,
            len(oos_labels),
        )
        return True, metrics

    # Evaluate current model on OOS
    current_filled = oos_features[current_model.feature_names].fillna(0.0)
    current_preds = current_model.pipeline.predict(current_filled)
    current_accuracy = float(np.mean(current_preds == oos_labels.values))
    metrics["current_oos_accuracy"] = current_accuracy

    if new_accuracy > current_accuracy:
        metrics["promoted"] = 1.0
        logger.info(
            "oos_gate result=promote new_oos_accuracy=%.4f "
            "current_oos_accuracy=%.4f oos_size=%d",
            new_accuracy,
            current_accuracy,
            len(oos_labels),
        )
        return True, metrics

    metrics["promoted"] = 0.0
    logger.info(
        "oos_gate result=blocked new_oos_accuracy=%.4f "
        "current_oos_accuracy=%.4f oos_size=%d",
        new_accuracy,
        current_accuracy,
        len(oos_labels),
    )
    return False, metrics


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

    # 4. Warn if training samples are less than 90% of available data
    available = len(dates) - 1
    if available > 0 and len(feature_rows) < 0.9 * available:
        logger.warning(
            "data_utilization_gap ticker=%s samples=%d available=%d "
            "ratio=%.2f",
            ticker,
            len(feature_rows),
            available,
            len(feature_rows) / available,
        )

    # 5. Train the model
    features_df = pd.DataFrame(feature_rows)
    labels_series = pd.Series(labels)
    model: TrainedModel = train_model(features_df, labels_series)

    # 6. Save with incremented version
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


def retrain_with_validation(
    conn: sqlite3.Connection,
    ticker: str,
    models_dir: str = "models",
    oos_days: int = 30,
) -> ModelRecord | None:
    """Retrain the model for *ticker* with OOS validation and rollback.

    Performs the full retrain cycle: builds features/labels from all
    available historical data, trains a new model, then validates it
    against the currently deployed model via the OOS gate.

    - If the new model passes the OOS gate, it is saved and promoted.
    - If the new model fails the OOS gate, it is **not** saved—the
      previous model version remains active and a rollback event is logged.

    Args:
        conn: SQLite connection with schema already applied.
        ticker: Stock ticker symbol.
        models_dir: Directory for model artifact storage.
        oos_days: Number of most-recent samples to hold out for OOS
            validation.

    Returns:
        :class:`ModelRecord` for the new model if promoted, or ``None``
        if the OOS gate blocked and a rollback occurred.

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

    # 4. Warn if training samples are less than 90% of available data
    available = len(dates) - 1
    if available > 0 and len(feature_rows) < 0.9 * available:
        logger.warning(
            "data_utilization_gap ticker=%s samples=%d available=%d "
            "ratio=%.2f",
            ticker,
            len(feature_rows),
            available,
            len(feature_rows) / available,
        )

    # 5. Train the new model
    features_df = pd.DataFrame(feature_rows)
    labels_series = pd.Series(labels)
    new_model: TrainedModel = train_model(features_df, labels_series)

    # 5. Load the current model for OOS comparison
    current_result = load_latest_model(conn, ticker)
    current_model = current_result[0] if current_result is not None else None

    # 6. OOS validation gate
    promoted, oos_metrics = validate_oos(
        new_model, features_df, labels_series,
        current_model=current_model, oos_days=oos_days,
    )

    elapsed = time.monotonic() - t0

    if promoted:
        # Save and promote the new model
        record = save_model(conn, ticker, new_model, models_dir=models_dir)
        logger.info(
            "retrain_complete ticker=%s version=%d accuracy=%.4f "
            "train_samples=%d elapsed=%.2fs",
            ticker,
            record.version,
            new_model.metrics.get("accuracy", 0.0),
            len(feature_rows),
            elapsed,
        )
        return record

    # 7. Rollback: do NOT save the new model; previous version stays active
    logger.warning(
        "rollback ticker=%s reason=oos_gate_failed "
        "new_oos_accuracy=%.4f current_oos_accuracy=%.4f "
        "oos_size=%d elapsed=%.2fs",
        ticker,
        oos_metrics.get("new_oos_accuracy", 0.0),
        oos_metrics.get("current_oos_accuracy", 0.0),
        int(oos_metrics.get("oos_size", 0)),
        elapsed,
    )
    return None


def detect_drift(
    conn: sqlite3.Connection,
    ticker: str,
    window_days: int = 30,
    p_threshold: float = 0.05,
    as_of_date: datetime.date | None = None,
    reports_dir: str = "reports",
) -> dict[str, object]:
    """Detect feature drift using the two-sample Kolmogorov–Smirnov test.

    Compares the distribution of each feature in the training period
    (all dates before the recent window) against the recent *window_days*
    trading days.  If the KS-test p-value for any feature falls below
    *p_threshold*, a WARNING log is emitted.

    A drift report is persisted to ``reports/drift_<date>.json``.

    Args:
        conn: SQLite connection with schema already applied.
        ticker: Stock ticker symbol.
        window_days: Number of most-recent trading days for the recent window.
        p_threshold: p-value below which drift is flagged.
        as_of_date: Date to anchor the analysis (defaults to today).
        reports_dir: Directory for drift report storage.

    Returns:
        A dict containing ``ticker``, ``as_of_date``, ``features`` (per-feature
        results), and ``drifted_features`` (list of feature names with drift).
    """
    if as_of_date is None:
        as_of_date = datetime.date.today()

    # 1. Get all trading dates up to as_of_date
    dates = _get_trading_dates(conn, ticker)
    dates = [d for d in dates if d <= as_of_date]

    if len(dates) <= window_days:
        logger.info(
            "drift_check ticker=%s result=skip reason=insufficient_data "
            "trading_days=%d window_days=%d",
            ticker,
            len(dates),
            window_days,
        )
        return {
            "ticker": ticker,
            "as_of_date": as_of_date.isoformat(),
            "features": {},
            "drifted_features": [],
            "skipped": True,
            "reason": "insufficient_data",
        }

    # 2. Split into training (older) and recent (last window_days) periods
    recent_dates = dates[-window_days:]
    training_dates = dates[:-window_days]

    if len(training_dates) == 0:
        logger.info(
            "drift_check ticker=%s result=skip reason=no_training_dates "
            "trading_days=%d window_days=%d",
            ticker,
            len(dates),
            window_days,
        )
        return {
            "ticker": ticker,
            "as_of_date": as_of_date.isoformat(),
            "features": {},
            "drifted_features": [],
            "skipped": True,
            "reason": "no_training_dates",
        }

    # 3. Build features for both periods
    training_features = [build_features(conn, ticker, dt) for dt in training_dates]
    recent_features = [build_features(conn, ticker, dt) for dt in recent_dates]

    training_df = pd.DataFrame(training_features)
    recent_df = pd.DataFrame(recent_features)

    # 4. KS-test on each feature
    feature_results: dict[str, dict[str, float]] = {}
    drifted: list[str] = []

    for feature_name in sorted(FEATURE_KEYS):
        train_vals = training_df[feature_name].dropna().values
        recent_vals = recent_df[feature_name].dropna().values

        if len(train_vals) < 2 or len(recent_vals) < 2:
            feature_results[feature_name] = {
                "statistic": float("nan"),
                "p_value": float("nan"),
                "drifted": 0.0,
            }
            continue

        stat, p_value = ks_2samp(train_vals, recent_vals)
        is_drifted = p_value < p_threshold

        feature_results[feature_name] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "drifted": 1.0 if is_drifted else 0.0,
        }

        if is_drifted:
            drifted.append(feature_name)
            logger.warning(
                "drift_alert ticker=%s feature=%s ks_statistic=%.4f "
                "p_value=%.6f threshold=%.4f",
                ticker,
                feature_name,
                stat,
                p_value,
                p_threshold,
            )

    # 5. Build and persist the report
    report: dict[str, object] = {
        "ticker": ticker,
        "as_of_date": as_of_date.isoformat(),
        "window_days": window_days,
        "p_threshold": p_threshold,
        "training_samples": len(training_dates),
        "recent_samples": len(recent_dates),
        "features": feature_results,
        "drifted_features": drifted,
    }

    report_path = Path(reports_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    filepath = report_path / f"drift_{as_of_date.isoformat()}.json"
    filepath.write_text(json.dumps(report, indent=2))

    if drifted:
        logger.warning(
            "drift_check ticker=%s result=drift_detected "
            "drifted_count=%d features=%s",
            ticker,
            len(drifted),
            ",".join(drifted),
        )
    else:
        logger.info(
            "drift_check ticker=%s result=no_drift "
            "features_checked=%d",
            ticker,
            len(feature_results),
        )

    return report
