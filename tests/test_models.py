"""Tests for SMAPS data models."""

from __future__ import annotations

import datetime

import pytest

from smaps.models import Direction, OHLCVBar, PredictionResult


class TestOHLCVBar:
    """Tests for OHLCVBar dataclass."""

    def test_create_valid_bar(self) -> None:
        """Valid OHLCVBar can be created with correct fields."""
        bar = OHLCVBar(
            ticker="AAPL",
            date=datetime.date(2025, 1, 15),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1_000_000,
        )
        assert bar.ticker == "AAPL"
        assert bar.date == datetime.date(2025, 1, 15)
        assert bar.open == 150.0
        assert bar.high == 155.0
        assert bar.low == 149.0
        assert bar.close == 153.0
        assert bar.volume == 1_000_000

    def test_create_bar_high_equals_low(self) -> None:
        """OHLCVBar allows high == low (flat day)."""
        bar = OHLCVBar(
            ticker="MSFT",
            date=datetime.date(2025, 1, 15),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=0,
        )
        assert bar.high == bar.low

    def test_validation_high_less_than_low(self) -> None:
        """OHLCVBar rejects high < low."""
        with pytest.raises(ValueError, match="high.*must be >= low"):
            OHLCVBar(
                ticker="AAPL",
                date=datetime.date(2025, 1, 15),
                open=150.0,
                high=148.0,
                low=149.0,
                close=149.0,
                volume=100,
            )

    def test_validation_negative_volume(self) -> None:
        """OHLCVBar rejects negative volume."""
        with pytest.raises(ValueError, match="volume.*must be >= 0"):
            OHLCVBar(
                ticker="AAPL",
                date=datetime.date(2025, 1, 15),
                open=150.0,
                high=155.0,
                low=149.0,
                close=153.0,
                volume=-1,
            )

    def test_bar_is_frozen(self) -> None:
        """OHLCVBar is immutable (frozen dataclass)."""
        bar = OHLCVBar(
            ticker="AAPL",
            date=datetime.date(2025, 1, 15),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1_000_000,
        )
        with pytest.raises(AttributeError):
            bar.close = 999.0  # type: ignore[misc]


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_create_valid_prediction(self) -> None:
        """Valid PredictionResult can be created with correct fields."""
        pred = PredictionResult(
            ticker="AAPL",
            prediction_date=datetime.date(2025, 1, 15),
            direction=Direction.UP,
            confidence=0.85,
            model_version="v1",
        )
        assert pred.ticker == "AAPL"
        assert pred.prediction_date == datetime.date(2025, 1, 15)
        assert pred.direction == Direction.UP
        assert pred.confidence == 0.85
        assert pred.model_version == "v1"

    def test_direction_down(self) -> None:
        """PredictionResult accepts Direction.DOWN."""
        pred = PredictionResult(
            ticker="MSFT",
            prediction_date=datetime.date(2025, 1, 15),
            direction=Direction.DOWN,
            confidence=0.60,
            model_version="v2",
        )
        assert pred.direction == Direction.DOWN
        assert pred.direction.value == "DOWN"

    def test_direction_enum_values(self) -> None:
        """Direction enum has exactly UP and DOWN members."""
        assert set(Direction) == {Direction.UP, Direction.DOWN}
        assert Direction.UP.value == "UP"
        assert Direction.DOWN.value == "DOWN"

    def test_confidence_boundary_zero(self) -> None:
        """PredictionResult accepts confidence = 0.0."""
        pred = PredictionResult(
            ticker="AAPL",
            prediction_date=datetime.date(2025, 1, 15),
            direction=Direction.DOWN,
            confidence=0.0,
            model_version="v1",
        )
        assert pred.confidence == 0.0

    def test_confidence_boundary_one(self) -> None:
        """PredictionResult accepts confidence = 1.0."""
        pred = PredictionResult(
            ticker="AAPL",
            prediction_date=datetime.date(2025, 1, 15),
            direction=Direction.UP,
            confidence=1.0,
            model_version="v1",
        )
        assert pred.confidence == 1.0

    def test_confidence_too_high(self) -> None:
        """PredictionResult rejects confidence > 1.0."""
        with pytest.raises(ValueError, match="confidence.*must be between 0.0 and 1.0"):
            PredictionResult(
                ticker="AAPL",
                prediction_date=datetime.date(2025, 1, 15),
                direction=Direction.UP,
                confidence=1.1,
                model_version="v1",
            )

    def test_confidence_negative(self) -> None:
        """PredictionResult rejects negative confidence."""
        with pytest.raises(ValueError, match="confidence.*must be between 0.0 and 1.0"):
            PredictionResult(
                ticker="AAPL",
                prediction_date=datetime.date(2025, 1, 15),
                direction=Direction.UP,
                confidence=-0.1,
                model_version="v1",
            )

    def test_prediction_is_frozen(self) -> None:
        """PredictionResult is immutable (frozen dataclass)."""
        pred = PredictionResult(
            ticker="AAPL",
            prediction_date=datetime.date(2025, 1, 15),
            direction=Direction.UP,
            confidence=0.85,
            model_version="v1",
        )
        with pytest.raises(AttributeError):
            pred.confidence = 0.5  # type: ignore[misc]
