"""Self-learning component: retrains the prediction model on mispredictions."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from .predictor import _build_features, _get_feature_cols
from .validator import ValidationResult

logger = logging.getLogger(__name__)

MODELS_DIR: Path = Path(__file__).parent.parent / "models"
HISTORY_FILE: Path = MODELS_DIR / "history.json"


@dataclass
class ModelVersion:
    version_id: str
    symbol: str
    trained_at: str       # ISO-8601 UTC timestamp
    triggered_by: str     # "misprediction" or "initial"
    training_samples: int
    model_path: str


def _load_history() -> list[dict[str, object]]:
    if HISTORY_FILE.exists():
        with HISTORY_FILE.open() as f:
            data: list[dict[str, object]] = json.load(f)
            return data
    return []


def _save_history(history: list[dict[str, object]]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("w") as f:
        json.dump(history, f, indent=2)


def retrain_model(symbol: str, triggered_by: str = "misprediction") -> ModelVersion:
    """Fetch fresh price data and retrain the logistic regression model for *symbol*.

    The fitted scaler and model are persisted to disk so subsequent calls can
    load a known-good version.  A record is appended to the model history file.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL').
        triggered_by: Human-readable reason for retraining.

    Returns:
        A :class:`ModelVersion` record describing the newly trained model.
    """
    logger.info("Retraining model for %s (reason: %s)", symbol, triggered_by)

    ticker = yf.Ticker(symbol)
    raw: pd.DataFrame = ticker.history(period="2y", auto_adjust=True)

    if raw.empty or len(raw) < 60:
        raise ValueError(f"Not enough price history for {symbol} to retrain")

    df = _build_features(raw)

    feature_cols = _get_feature_cols()
    X: np.ndarray = df[feature_cols].to_numpy()
    y: np.ndarray = df["target"].to_numpy()

    # Reserve last row (live prediction row — no next-day close yet); train on prior rows.
    X_train, y_train = X[:-1], y[:-1]

    scaler: StandardScaler = StandardScaler()
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)

    model: LogisticRegression = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version_id = uuid.uuid4().hex[:8]
    model_path = str(MODELS_DIR / f"{symbol}_{version_id}.joblib")
    joblib.dump({"scaler": scaler, "model": model}, model_path)

    trained_at = datetime.now(tz=timezone.utc).isoformat()
    version = ModelVersion(
        version_id=version_id,
        symbol=symbol,
        trained_at=trained_at,
        triggered_by=triggered_by,
        training_samples=int(len(X_train)),
        model_path=model_path,
    )

    history = _load_history()
    history.append(asdict(version))
    _save_history(history)

    logger.info(
        "Model retrained for %s: version=%s samples=%d path=%s",
        symbol,
        version_id,
        len(X_train),
        model_path,
    )
    return version


def learn_from_result(result: ValidationResult) -> ModelVersion | None:
    """Inspect a validation result and retrain the model if the prediction was wrong.

    This is the autonomous learning entry point: it fires without human
    intervention whenever ``result.correct`` is ``False``.

    Args:
        result: The :class:`ValidationResult` to learn from.

    Returns:
        A :class:`ModelVersion` if retraining occurred, or ``None`` if the
        prediction was correct and no update was needed.
    """
    if result.correct:
        logger.info(
            "%s prediction on %s was correct — no retraining needed",
            result.prediction.symbol,
            result.prediction.prediction_date,
        )
        return None

    logger.info(
        "%s prediction on %s was WRONG (predicted %s, actual %s) — triggering retraining",
        result.prediction.symbol,
        result.prediction.prediction_date,
        result.prediction.label,
        result.actual_label,
    )
    return retrain_model(result.prediction.symbol, triggered_by="misprediction")


def run_daily_learning(result: ValidationResult) -> ModelVersion | None:
    """Entry point for the daily learning job.

    Should be invoked after :func:`~src.validator.run_daily_validation`.

    Args:
        result: The :class:`ValidationResult` from the validation step.

    Returns:
        A :class:`ModelVersion` if the model was updated, else ``None``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return learn_from_result(result)
