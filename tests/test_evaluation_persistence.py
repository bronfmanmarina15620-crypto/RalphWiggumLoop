"""Tests for evaluation persistence (US-403)."""

from __future__ import annotations

import datetime
import json

from smaps.db import (
    EvalRecord,
    ensure_schema,
    get_connection,
    load_evaluation,
    save_evaluation,
    save_metrics_report,
    save_prediction,
)
from smaps.models import Direction, EvalResult, MetricsReport, PredictionResult


def _setup_db():
    conn = get_connection(":memory:")
    ensure_schema(conn)
    return conn


def _make_prediction(
    ticker: str = "AAPL",
    date: datetime.date | None = None,
    direction: Direction = Direction.UP,
) -> PredictionResult:
    return PredictionResult(
        ticker=ticker,
        prediction_date=date or datetime.date(2025, 1, 15),
        direction=direction,
        confidence=0.75,
        model_version="v1",
    )


def _make_eval_result(
    prediction_id: int = 1,
    actual_direction: Direction = Direction.UP,
    is_correct: bool = True,
) -> EvalResult:
    return EvalResult(
        prediction_id=prediction_id,
        actual_direction=actual_direction,
        is_correct=is_correct,
        evaluated_at=datetime.datetime(2025, 1, 16, 12, 0, 0, tzinfo=datetime.timezone.utc),
    )


class TestSaveEvaluation:
    def test_save_returns_autoincrement_id(self) -> None:
        conn = _setup_db()
        pred_id = save_prediction(conn, _make_prediction())
        result = _make_eval_result(prediction_id=pred_id)
        eval_id = save_evaluation(conn, result)
        assert isinstance(eval_id, int)
        assert eval_id >= 1

    def test_save_multiple_returns_sequential_ids(self) -> None:
        conn = _setup_db()
        pred_id1 = save_prediction(conn, _make_prediction(ticker="AAPL"))
        pred_id2 = save_prediction(conn, _make_prediction(ticker="MSFT"))
        id1 = save_evaluation(conn, _make_eval_result(prediction_id=pred_id1))
        id2 = save_evaluation(conn, _make_eval_result(prediction_id=pred_id2))
        assert id2 > id1

    def test_save_correct_evaluation(self) -> None:
        conn = _setup_db()
        pred_id = save_prediction(conn, _make_prediction())
        result = _make_eval_result(prediction_id=pred_id, is_correct=True)
        eval_id = save_evaluation(conn, result)
        loaded = load_evaluation(conn, eval_id)
        assert loaded is not None
        assert loaded.is_correct is True

    def test_save_incorrect_evaluation(self) -> None:
        conn = _setup_db()
        pred_id = save_prediction(conn, _make_prediction())
        result = _make_eval_result(
            prediction_id=pred_id,
            actual_direction=Direction.DOWN,
            is_correct=False,
        )
        eval_id = save_evaluation(conn, result)
        loaded = load_evaluation(conn, eval_id)
        assert loaded is not None
        assert loaded.is_correct is False


class TestLoadEvaluation:
    def test_round_trip(self) -> None:
        """Core AC: save and load evaluation, verify all fields match."""
        conn = _setup_db()
        pred_id = save_prediction(conn, _make_prediction())
        result = EvalResult(
            prediction_id=pred_id,
            actual_direction=Direction.DOWN,
            is_correct=False,
            evaluated_at=datetime.datetime(2025, 2, 20, 8, 30, 0, tzinfo=datetime.timezone.utc),
        )
        eval_id = save_evaluation(conn, result)
        loaded = load_evaluation(conn, eval_id)

        assert loaded is not None
        assert isinstance(loaded, EvalRecord)
        assert loaded.id == eval_id
        assert loaded.prediction_id == pred_id
        assert loaded.actual_direction == Direction.DOWN
        assert loaded.is_correct is False
        assert loaded.evaluated_at  # non-empty ISO datetime string

    def test_load_nonexistent_returns_none(self) -> None:
        conn = _setup_db()
        assert load_evaluation(conn, 9999) is None

    def test_direction_up_round_trip(self) -> None:
        conn = _setup_db()
        pred_id = save_prediction(conn, _make_prediction())
        result = _make_eval_result(prediction_id=pred_id, actual_direction=Direction.UP)
        eval_id = save_evaluation(conn, result)
        loaded = load_evaluation(conn, eval_id)
        assert loaded is not None
        assert loaded.actual_direction == Direction.UP

    def test_direction_down_round_trip(self) -> None:
        conn = _setup_db()
        pred_id = save_prediction(conn, _make_prediction())
        result = _make_eval_result(prediction_id=pred_id, actual_direction=Direction.DOWN)
        eval_id = save_evaluation(conn, result)
        loaded = load_evaluation(conn, eval_id)
        assert loaded is not None
        assert loaded.actual_direction == Direction.DOWN

    def test_evaluated_at_is_iso_datetime(self) -> None:
        conn = _setup_db()
        pred_id = save_prediction(conn, _make_prediction())
        result = _make_eval_result(prediction_id=pred_id)
        eval_id = save_evaluation(conn, result)
        loaded = load_evaluation(conn, eval_id)
        assert loaded is not None
        dt = datetime.datetime.fromisoformat(loaded.evaluated_at)
        assert dt.tzinfo is not None  # UTC timezone present


class TestSaveMetricsReport:
    def test_creates_report_file(self, tmp_path) -> None:
        report = MetricsReport(
            ticker="AAPL",
            window_start=datetime.date(2024, 10, 1),
            window_end=datetime.date(2025, 1, 1),
            accuracy=0.65,
            precision_up=0.7,
            recall_up=0.6,
            precision_down=0.6,
            recall_down=0.7,
            total_predictions=20,
            evaluated_predictions=18,
        ).to_dict()
        filepath = save_metrics_report(report, reports_dir=str(tmp_path / "reports"))
        assert filepath.exists()
        assert filepath.name == "metrics_AAPL_2025-01-01.json"

    def test_report_content_is_valid_json(self, tmp_path) -> None:
        report = MetricsReport(
            ticker="MSFT",
            window_start=datetime.date(2024, 10, 1),
            window_end=datetime.date(2025, 1, 1),
            accuracy=0.55,
            precision_up=0.5,
            recall_up=0.6,
            precision_down=0.6,
            recall_down=0.5,
            total_predictions=10,
            evaluated_predictions=8,
        ).to_dict()
        filepath = save_metrics_report(report, reports_dir=str(tmp_path / "reports"))
        loaded = json.loads(filepath.read_text())
        assert loaded["ticker"] == "MSFT"
        assert loaded["accuracy"] == 0.55
        assert loaded["total_predictions"] == 10

    def test_creates_reports_directory(self, tmp_path) -> None:
        reports_dir = tmp_path / "new" / "reports"
        assert not reports_dir.exists()
        report = MetricsReport(
            ticker="GOOG",
            window_start=datetime.date(2024, 10, 1),
            window_end=datetime.date(2025, 1, 1),
            accuracy=0.5,
            precision_up=0.5,
            recall_up=0.5,
            precision_down=0.5,
            recall_down=0.5,
            total_predictions=5,
            evaluated_predictions=5,
        ).to_dict()
        filepath = save_metrics_report(report, reports_dir=str(reports_dir))
        assert reports_dir.exists()
        assert filepath.exists()

    def test_overwrites_existing_report(self, tmp_path) -> None:
        report1 = {"ticker": "AAPL", "window_end": "2025-01-01", "accuracy": 0.5}
        report2 = {"ticker": "AAPL", "window_end": "2025-01-01", "accuracy": 0.8}
        reports_dir = str(tmp_path / "reports")
        save_metrics_report(report1, reports_dir=reports_dir)
        filepath = save_metrics_report(report2, reports_dir=reports_dir)
        loaded = json.loads(filepath.read_text())
        assert loaded["accuracy"] == 0.8
