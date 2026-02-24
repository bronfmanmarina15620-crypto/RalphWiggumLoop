"""Prediction outcome validator: checks if next-day movement matched the prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from .predictor import Prediction

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    prediction: Prediction
    actual_label: str        # "UP" or "DOWN"
    correct: bool
    prediction_close: float  # closing price on prediction_date (or nearest prior trading day)
    next_close: float        # closing price on next trading day


def validate_prediction(prediction: Prediction) -> ValidationResult:
    """Fetch actual price data and evaluate whether the prediction was correct.

    Args:
        prediction: The :class:`Prediction` to validate.

    Returns:
        A :class:`ValidationResult` with actual movement and correctness flag.

    Raises:
        ValueError: If price data is unavailable for the required dates.
    """
    logger.info(
        "Validating %s prediction for %s",
        prediction.symbol,
        prediction.prediction_date,
    )

    # Fetch a window around the prediction date to capture actual prices.
    start = prediction.prediction_date - timedelta(days=5)
    end = prediction.prediction_date + timedelta(days=10)

    ticker = yf.Ticker(prediction.symbol)
    raw: pd.DataFrame = ticker.history(
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
    )

    if raw.empty:
        raise ValueError(
            f"No price data for {prediction.symbol} around {prediction.prediction_date}"
        )

    # Build a date -> close price mapping for straightforward comparison.
    close_series: pd.Series = raw["Close"].copy()
    close_series.index = pd.to_datetime(close_series.index).normalize()
    datetime_index: pd.DatetimeIndex = pd.DatetimeIndex(close_series.index)
    close_by_date: dict[date, float] = {
        ts.date(): float(close_series.iloc[i])
        for i, ts in enumerate(datetime_index)
    }
    available_dates: list[date] = sorted(close_by_date.keys())

    # The prediction may have been made on a non-trading day; find nearest prior day.
    pred_dates = [d for d in available_dates if d <= prediction.prediction_date]
    if not pred_dates:
        raise ValueError(
            f"No trading data on or before {prediction.prediction_date} for {prediction.symbol}"
        )
    actual_pred_date = pred_dates[-1]

    # Next trading day after the reference date.
    next_dates = [d for d in available_dates if d > actual_pred_date]
    if not next_dates:
        raise ValueError(
            f"No next-day trading data after {actual_pred_date} for {prediction.symbol}"
        )
    next_date = next_dates[0]

    prediction_close = close_by_date[actual_pred_date]
    next_close = close_by_date[next_date]

    actual_label = "UP" if next_close > prediction_close else "DOWN"
    correct = actual_label == prediction.label

    logger.info(
        "%s validation: predicted=%s actual=%s correct=%s (close %.2f -> %.2f)",
        prediction.symbol,
        prediction.label,
        actual_label,
        correct,
        prediction_close,
        next_close,
    )

    return ValidationResult(
        prediction=prediction,
        actual_label=actual_label,
        correct=correct,
        prediction_close=prediction_close,
        next_close=next_close,
    )


def run_daily_validation(prediction: Prediction) -> ValidationResult:
    """Entry point for the daily validation job.

    Should be invoked after market close each trading day.

    Args:
        prediction: The :class:`Prediction` to validate.

    Returns:
        The :class:`ValidationResult`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return validate_prediction(prediction)
