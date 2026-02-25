"""Technical indicator features computed from OHLCV data."""

from __future__ import annotations

import datetime
import math
import sqlite3


class TechnicalFeatures:
    """Compute price-derived technical features from OHLCV bars.

    Implements the ``FeaturePipeline`` protocol.  Only uses bars dated
    ``<= as_of_date`` to prevent look-ahead bias.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def transform(
        self, ticker: str, as_of_date: datetime.date
    ) -> dict[str, float]:
        """Return technical indicator features for *ticker* as of *as_of_date*."""
        closes, volumes = self._load_bars(ticker, as_of_date)

        return {
            "return_1d": self._return_n(closes, 1),
            "return_5d": self._return_n(closes, 5),
            "return_10d": self._return_n(closes, 10),
            "ma_ratio_5_20": self._ma_ratio(closes, 5, 20),
            "volume_change_1d": self._return_n(volumes, 1),
            "volatility_20d": self._volatility(closes, 20),
            "rsi_14": self._rsi(closes, 14),
        }

    def _load_bars(
        self, ticker: str, as_of_date: datetime.date
    ) -> tuple[list[float], list[float]]:
        """Load close prices and volumes up to *as_of_date* (inclusive)."""
        cur = self._conn.execute(
            "SELECT close, volume FROM ohlcv_daily "
            "WHERE ticker = ? AND date <= ? ORDER BY date ASC",
            (ticker, as_of_date.isoformat()),
        )
        rows = cur.fetchall()
        closes = [float(r[0]) for r in rows]
        volumes = [float(r[1]) for r in rows]
        return closes, volumes

    @staticmethod
    def _return_n(values: list[float], n: int) -> float:
        """N-period return: values[-1] / values[-1-n] - 1."""
        if len(values) < n + 1 or values[-1 - n] == 0.0:
            return float("nan")
        return values[-1] / values[-1 - n] - 1.0

    @staticmethod
    def _ma_ratio(values: list[float], short: int, long: int) -> float:
        """Ratio of short-period MA to long-period MA."""
        if len(values) < long:
            return float("nan")
        ma_short = sum(values[-short:]) / short
        ma_long = sum(values[-long:]) / long
        if ma_long == 0.0:
            return float("nan")
        return ma_short / ma_long

    @staticmethod
    def _volatility(closes: list[float], window: int) -> float:
        """Standard deviation of daily returns over *window* days."""
        if len(closes) < window + 1:
            return float("nan")
        returns = [
            closes[i] / closes[i - 1] - 1.0
            for i in range(len(closes) - window, len(closes))
            if closes[i - 1] != 0.0
        ]
        if len(returns) < 2:
            return float("nan")
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(variance)

    @staticmethod
    def _rsi(closes: list[float], period: int) -> float:
        """Relative Strength Index over *period* days."""
        if len(closes) < period + 1:
            return float("nan")
        changes = [
            closes[i] - closes[i - 1]
            for i in range(len(closes) - period, len(closes))
        ]
        gains = [c for c in changes if c > 0]
        losses = [-c for c in changes if c < 0]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
