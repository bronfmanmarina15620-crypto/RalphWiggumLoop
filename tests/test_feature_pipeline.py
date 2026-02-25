"""Tests for the FeaturePipeline protocol (US-201)."""

from __future__ import annotations

import datetime

from smaps.features.pipeline import FeaturePipeline


class _DummyPipeline:
    """A concrete implementation used to verify the protocol."""

    def transform(
        self, ticker: str, as_of_date: datetime.date
    ) -> dict[str, float]:
        return {"return_1d": 0.01, "volatility_20d": 0.15}


def test_dummy_implements_protocol() -> None:
    """A class with the correct signature satisfies FeaturePipeline."""
    pipeline: FeaturePipeline = _DummyPipeline()
    result = pipeline.transform("AAPL", datetime.date(2025, 1, 15))
    assert isinstance(result, dict)
    assert "return_1d" in result
    assert "volatility_20d" in result


def test_transform_returns_dict_of_floats() -> None:
    """transform() returns dict[str, float]."""
    pipeline: FeaturePipeline = _DummyPipeline()
    result = pipeline.transform("MSFT", datetime.date(2025, 6, 1))
    for key, value in result.items():
        assert isinstance(key, str)
        assert isinstance(value, float)


def test_protocol_is_runtime_checkable_via_typing() -> None:
    """FeaturePipeline is a Protocol and can be used as a type annotation."""
    # Structural subtyping: _DummyPipeline has the right shape.
    pipeline = _DummyPipeline()
    # mypy will verify this assignment at typecheck time; at runtime we just
    # confirm the object has the transform method.
    assert hasattr(pipeline, "transform")
    assert callable(pipeline.transform)
