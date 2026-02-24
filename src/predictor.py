"""Stock movement predictor: predicts whether a stock will rise or fall next day."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

LOOKBACK_DAYS: int = 252  # ~1 trading year for feature computation


@dataclass
class Prediction:
    symbol: str
    prediction_date: date
    label: str          # "UP" or "DOWN"
    up_probability: float
    down_probability: float


def _build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicator features from OHLCV price data."""
    df = prices[["Close", "Volume"]].copy()
    close = df["Close"]

    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)

    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma5_ma20_ratio"] = df["ma5"] / df["ma20"]

    df["vol_change_1d"] = df["Volume"].pct_change(1)

    rolling_std = close.rolling(20).std()
    df["volatility_20d"] = rolling_std / close

    df["target"] = (close.shift(-1) > close).astype(int)

    df.dropna(inplace=True)
    return df


def _get_feature_cols() -> list[str]:
    return [
        "return_1d",
        "return_5d",
        "return_10d",
        "ma5_ma20_ratio",
        "vol_change_1d",
        "volatility_20d",
    ]


def predict_movement(symbol: str) -> Prediction:
    """Fetch historical data, train a model, and predict tomorrow's direction.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL').

    Returns:
        A :class:`Prediction` with label and probabilities.
    """
    logger.info("Fetching price history for %s", symbol)
    ticker = yf.Ticker(symbol)
    raw: pd.DataFrame = ticker.history(period="2y", auto_adjust=True)

    if raw.empty or len(raw) < 60:
        raise ValueError(f"Not enough price history for {symbol}")

    df = _build_features(raw)

    feature_cols = _get_feature_cols()
    X: np.ndarray = df[feature_cols].to_numpy()
    y: np.ndarray = df["target"].to_numpy()

    # Reserve last row for prediction (no future label yet)
    X_train, y_train = X[:-1], y[:-1]
    X_pred = X[-1].reshape(1, -1)

    scaler = StandardScaler()
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
    X_pred_scaled: np.ndarray = scaler.transform(X_pred)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    proba: np.ndarray = model.predict_proba(X_pred_scaled)[0]
    up_prob = float(proba[1])
    down_prob = float(proba[0])
    label = "UP" if up_prob >= 0.5 else "DOWN"

    prediction_date = datetime.now(tz=timezone.utc).date()

    logger.info(
        "%s prediction for %s: %s (up=%.2f%%, down=%.2f%%)",
        symbol,
        prediction_date,
        label,
        up_prob * 100,
        down_prob * 100,
    )

    return Prediction(
        symbol=symbol,
        prediction_date=prediction_date,
        label=label,
        up_probability=up_prob,
        down_probability=down_prob,
    )


def run_daily_prediction(symbol: str) -> Prediction:
    """Entry point for the daily prediction job.

    Intended to be invoked once per trading day (e.g. by a scheduler).

    Args:
        symbol: Ticker symbol to predict.

    Returns:
        The :class:`Prediction` result.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return predict_movement(symbol)


if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = run_daily_prediction(sym)
    print(
        f"{result.symbol} | {result.prediction_date} | {result.label} "
        f"| up={result.up_probability:.2%} down={result.down_probability:.2%}"
    )
