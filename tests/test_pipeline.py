"""Tests for the daily pipeline orchestrator."""

from __future__ import annotations

import datetime
import logging
from unittest.mock import patch

import pytest

from smaps.models import Direction, Fundamentals, PredictionResult, SentimentScore
from smaps.pipeline import run_pipeline


DATE = datetime.date(2025, 6, 15)


def _make_prediction(ticker: str = "AAPL") -> PredictionResult:
    return PredictionResult(
        ticker=ticker,
        prediction_date=DATE,
        direction=Direction.UP,
        confidence=0.65,
        model_version="v1",
    )


def _make_sentiment(ticker: str = "AAPL") -> SentimentScore:
    return SentimentScore(
        ticker=ticker,
        date=DATE,
        score=0.1,
        source="google_news_rss",
    )


def _make_fundamentals(ticker: str = "AAPL") -> Fundamentals:
    return Fundamentals(
        ticker=ticker,
        date=DATE,
        pe_ratio=25.0,
        market_cap=3e12,
        eps=6.5,
        revenue=400e9,
    )


# ---------------------------------------------------------------------------
# Shared decorator stack to mock all external dependencies
# ---------------------------------------------------------------------------
def _patch_all():
    """Return a tuple of decorators that mock all pipeline external calls."""
    return (
        patch("smaps.pipeline.should_retrain", return_value=False),
        patch("smaps.pipeline.predict"),
        patch("smaps.pipeline.fetch_fundamentals"),
        patch("smaps.pipeline.fetch_sentiment"),
        patch("smaps.pipeline.fetch_daily_bars", return_value=[]),
    )


class TestRunPipeline:
    """Tests for run_pipeline call sequence and result structure."""

    @patch("smaps.pipeline.should_retrain", return_value=False)
    @patch("smaps.pipeline.predict")
    @patch("smaps.pipeline.fetch_fundamentals")
    @patch("smaps.pipeline.fetch_sentiment")
    @patch("smaps.pipeline.fetch_daily_bars", return_value=[])
    def test_calls_all_steps_for_ticker(
        self, mock_bars, mock_sent, mock_fund, mock_predict, mock_retrain
    ):
        """Verify ingest, predict, and retrain-check are called for each ticker."""
        mock_sent.return_value = _make_sentiment()
        mock_fund.return_value = _make_fundamentals()
        mock_predict.return_value = _make_prediction()

        run_pipeline(["AAPL"], DATE)

        mock_bars.assert_called_once()
        mock_sent.assert_called_once()
        mock_fund.assert_called_once()
        mock_predict.assert_called_once()
        mock_retrain.assert_called_once()

    @patch("smaps.pipeline.should_retrain", return_value=False)
    @patch("smaps.pipeline.predict")
    @patch("smaps.pipeline.fetch_fundamentals")
    @patch("smaps.pipeline.fetch_sentiment")
    @patch("smaps.pipeline.fetch_daily_bars", return_value=[])
    def test_call_sequence_order(
        self, mock_bars, mock_sent, mock_fund, mock_predict, mock_retrain
    ):
        """Verify steps run in order: ingest → predict → retrain-check."""
        call_order: list[str] = []

        def bars_effect(*args, **kwargs):
            call_order.append("ingest")
            return []

        def predict_effect(*args, **kwargs):
            call_order.append("predict")
            return _make_prediction()

        def retrain_effect(*args, **kwargs):
            call_order.append("retrain_check")
            return False

        mock_bars.side_effect = bars_effect
        mock_sent.return_value = _make_sentiment()
        mock_fund.return_value = _make_fundamentals()
        mock_predict.side_effect = predict_effect
        mock_retrain.side_effect = retrain_effect

        run_pipeline(["AAPL"], DATE)

        assert call_order == ["ingest", "predict", "retrain_check"]

    @patch("smaps.pipeline.should_retrain", return_value=False)
    @patch("smaps.pipeline.predict")
    @patch("smaps.pipeline.fetch_fundamentals")
    @patch("smaps.pipeline.fetch_sentiment")
    @patch("smaps.pipeline.fetch_daily_bars", return_value=[])
    def test_runs_all_tickers(
        self, mock_bars, mock_sent, mock_fund, mock_predict, mock_retrain
    ):
        """Verify all tickers in the list are processed."""
        mock_sent.side_effect = [_make_sentiment("AAPL"), _make_sentiment("MSFT")]
        mock_fund.side_effect = [
            _make_fundamentals("AAPL"),
            _make_fundamentals("MSFT"),
        ]
        mock_predict.side_effect = [
            _make_prediction("AAPL"),
            _make_prediction("MSFT"),
        ]

        result = run_pipeline(["AAPL", "MSFT"], DATE)

        assert mock_bars.call_count == 2
        assert mock_predict.call_count == 2
        assert "AAPL" in result["tickers"]
        assert "MSFT" in result["tickers"]


class TestFailureIsolation:
    """Tests for error isolation between tickers."""

    @patch("smaps.pipeline.should_retrain", return_value=False)
    @patch("smaps.pipeline.predict")
    @patch("smaps.pipeline.fetch_fundamentals")
    @patch("smaps.pipeline.fetch_sentiment")
    @patch("smaps.pipeline.fetch_daily_bars")
    def test_error_in_one_ticker_does_not_block_others(
        self, mock_bars, mock_sent, mock_fund, mock_predict, mock_retrain
    ):
        """Ingest failure for AAPL should not prevent MSFT from running."""

        def bars_effect(ticker, *args, **kwargs):
            if ticker == "AAPL":
                raise RuntimeError("API error")
            return []

        def predict_effect(conn, ticker, *args, **kwargs):
            if ticker == "AAPL":
                raise RuntimeError("No model")
            return _make_prediction(ticker)

        mock_bars.side_effect = bars_effect
        mock_sent.return_value = _make_sentiment("MSFT")
        mock_fund.return_value = _make_fundamentals("MSFT")
        mock_predict.side_effect = predict_effect

        result = run_pipeline(["AAPL", "MSFT"], DATE)

        tickers = result["tickers"]
        assert tickers["AAPL"]["ingest"] == "error"
        assert tickers["AAPL"]["predict"] == "error"
        assert tickers["MSFT"]["ingest"] == "ok"
        assert tickers["MSFT"]["predict"] == "ok"

    @patch("smaps.pipeline.should_retrain", return_value=False)
    @patch("smaps.pipeline.predict")
    @patch("smaps.pipeline.fetch_fundamentals")
    @patch("smaps.pipeline.fetch_sentiment")
    @patch("smaps.pipeline.fetch_daily_bars", return_value=[])
    def test_predict_error_does_not_block_evaluate(
        self, mock_bars, mock_sent, mock_fund, mock_predict, mock_retrain
    ):
        """Predict failure should not prevent the evaluate step."""
        mock_sent.return_value = _make_sentiment()
        mock_fund.return_value = _make_fundamentals()
        mock_predict.side_effect = RuntimeError("No model")

        result = run_pipeline(["AAPL"], DATE)

        tickers = result["tickers"]
        assert tickers["AAPL"]["ingest"] == "ok"
        assert tickers["AAPL"]["predict"] == "error"
        assert tickers["AAPL"]["evaluate"] == "ok"


class TestResultStructure:
    """Tests for the result dict returned by run_pipeline."""

    @patch("smaps.pipeline.should_retrain", return_value=False)
    @patch("smaps.pipeline.predict")
    @patch("smaps.pipeline.fetch_fundamentals")
    @patch("smaps.pipeline.fetch_sentiment")
    @patch("smaps.pipeline.fetch_daily_bars", return_value=[])
    def test_result_contains_run_id(
        self, mock_bars, mock_sent, mock_fund, mock_predict, mock_retrain
    ):
        mock_sent.return_value = _make_sentiment()
        mock_fund.return_value = _make_fundamentals()
        mock_predict.return_value = _make_prediction()

        result = run_pipeline(["AAPL"], DATE)

        assert "run_id" in result
        assert isinstance(result["run_id"], str)
        assert len(result["run_id"]) == 8

    @patch("smaps.pipeline.should_retrain", return_value=False)
    @patch("smaps.pipeline.predict")
    @patch("smaps.pipeline.fetch_fundamentals")
    @patch("smaps.pipeline.fetch_sentiment")
    @patch("smaps.pipeline.fetch_daily_bars", return_value=[])
    def test_result_contains_date_and_elapsed(
        self, mock_bars, mock_sent, mock_fund, mock_predict, mock_retrain
    ):
        mock_sent.return_value = _make_sentiment()
        mock_fund.return_value = _make_fundamentals()
        mock_predict.return_value = _make_prediction()

        result = run_pipeline(["AAPL"], DATE)

        assert result["date"] == DATE.isoformat()
        assert "elapsed" in result
        assert isinstance(result["elapsed"], float)
        assert result["elapsed"] >= 0

    @patch("smaps.pipeline.should_retrain", return_value=False)
    @patch("smaps.pipeline.predict")
    @patch("smaps.pipeline.fetch_fundamentals")
    @patch("smaps.pipeline.fetch_sentiment")
    @patch("smaps.pipeline.fetch_daily_bars", return_value=[])
    def test_ticker_result_has_all_step_keys(
        self, mock_bars, mock_sent, mock_fund, mock_predict, mock_retrain
    ):
        mock_sent.return_value = _make_sentiment()
        mock_fund.return_value = _make_fundamentals()
        mock_predict.return_value = _make_prediction()

        result = run_pipeline(["AAPL"], DATE)

        ticker_result = result["tickers"]["AAPL"]
        assert "ingest" in ticker_result
        assert "predict" in ticker_result
        assert "evaluate" in ticker_result
        assert "retrain" in ticker_result

    @patch("smaps.pipeline.should_retrain", return_value=False)
    @patch("smaps.pipeline.predict")
    @patch("smaps.pipeline.fetch_fundamentals")
    @patch("smaps.pipeline.fetch_sentiment")
    @patch("smaps.pipeline.fetch_daily_bars", return_value=[])
    def test_empty_ticker_list(
        self, mock_bars, mock_sent, mock_fund, mock_predict, mock_retrain
    ):
        result = run_pipeline([], DATE)

        assert result["tickers"] == {}
        mock_bars.assert_not_called()
        mock_predict.assert_not_called()


class TestRetrainIntegration:
    """Tests for the retrain step within the pipeline."""

    @patch("smaps.pipeline.retrain_with_validation")
    @patch("smaps.pipeline.should_retrain", return_value=True)
    @patch("smaps.pipeline.predict")
    @patch("smaps.pipeline.fetch_fundamentals")
    @patch("smaps.pipeline.fetch_sentiment")
    @patch("smaps.pipeline.fetch_daily_bars", return_value=[])
    def test_retrain_triggered_when_should_retrain_true(
        self,
        mock_bars,
        mock_sent,
        mock_fund,
        mock_predict,
        mock_retrain_check,
        mock_retrain_exec,
    ):
        """Retrain is called when should_retrain returns True."""
        from smaps.model.registry import ModelRecord

        mock_sent.return_value = _make_sentiment()
        mock_fund.return_value = _make_fundamentals()
        mock_predict.return_value = _make_prediction()
        mock_retrain_exec.return_value = ModelRecord(
            id=1, ticker="AAPL", version=2,
            trained_at="2025-06-15T00:00:00+00:00",
            metrics_json="{}", artifact_path="models/AAPL_v2.joblib",
        )

        result = run_pipeline(["AAPL"], DATE)

        mock_retrain_exec.assert_called_once()
        assert result["tickers"]["AAPL"]["retrain"] == "retrained"

    @patch("smaps.pipeline.retrain_with_validation", return_value=None)
    @patch("smaps.pipeline.should_retrain", return_value=True)
    @patch("smaps.pipeline.predict")
    @patch("smaps.pipeline.fetch_fundamentals")
    @patch("smaps.pipeline.fetch_sentiment")
    @patch("smaps.pipeline.fetch_daily_bars", return_value=[])
    def test_rollback_status_when_oos_gate_fails(
        self,
        mock_bars,
        mock_sent,
        mock_fund,
        mock_predict,
        mock_retrain_check,
        mock_retrain_exec,
    ):
        """Result shows 'rollback' when retrain_with_validation returns None."""
        mock_sent.return_value = _make_sentiment()
        mock_fund.return_value = _make_fundamentals()
        mock_predict.return_value = _make_prediction()

        result = run_pipeline(["AAPL"], DATE)

        assert result["tickers"]["AAPL"]["retrain"] == "rollback"

    @patch("smaps.pipeline.should_retrain", return_value=False)
    @patch("smaps.pipeline.predict")
    @patch("smaps.pipeline.fetch_fundamentals")
    @patch("smaps.pipeline.fetch_sentiment")
    @patch("smaps.pipeline.fetch_daily_bars", return_value=[])
    def test_retrain_not_needed(
        self, mock_bars, mock_sent, mock_fund, mock_predict, mock_retrain
    ):
        mock_sent.return_value = _make_sentiment()
        mock_fund.return_value = _make_fundamentals()
        mock_predict.return_value = _make_prediction()

        result = run_pipeline(["AAPL"], DATE)

        assert result["tickers"]["AAPL"]["retrain"] == "not_needed"


class TestPipelineLogging:
    """Tests for structured logging in the pipeline."""

    @patch("smaps.pipeline.should_retrain", return_value=False)
    @patch("smaps.pipeline.predict")
    @patch("smaps.pipeline.fetch_fundamentals")
    @patch("smaps.pipeline.fetch_sentiment")
    @patch("smaps.pipeline.fetch_daily_bars", return_value=[])
    def test_logs_pipeline_start_and_complete(
        self, mock_bars, mock_sent, mock_fund, mock_predict, mock_retrain, caplog
    ):
        mock_sent.return_value = _make_sentiment()
        mock_fund.return_value = _make_fundamentals()
        mock_predict.return_value = _make_prediction()

        with caplog.at_level(logging.INFO, logger="smaps.pipeline"):
            run_pipeline(["AAPL"], DATE)

        messages = [r.message for r in caplog.records]
        assert any("pipeline_start" in m for m in messages)
        assert any("pipeline_complete" in m for m in messages)

    @patch("smaps.pipeline.should_retrain", return_value=False)
    @patch("smaps.pipeline.predict")
    @patch("smaps.pipeline.fetch_fundamentals")
    @patch("smaps.pipeline.fetch_sentiment")
    @patch("smaps.pipeline.fetch_daily_bars", return_value=[])
    def test_logs_each_step_with_elapsed(
        self, mock_bars, mock_sent, mock_fund, mock_predict, mock_retrain, caplog
    ):
        mock_sent.return_value = _make_sentiment()
        mock_fund.return_value = _make_fundamentals()
        mock_predict.return_value = _make_prediction()

        with caplog.at_level(logging.INFO, logger="smaps.pipeline"):
            run_pipeline(["AAPL"], DATE)

        step_messages = [
            r.message for r in caplog.records if "step=" in r.message
        ]
        assert len(step_messages) >= 4  # ingest, predict, evaluate, retrain
        for msg in step_messages:
            assert "elapsed=" in msg
