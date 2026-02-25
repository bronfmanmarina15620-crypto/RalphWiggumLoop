"""Tests for US-303: model artifact persistence and versioned registry."""

from __future__ import annotations

import datetime
import json
import os

import numpy as np
import pandas as pd
import pytest

from smaps.db import ensure_schema, get_connection
from smaps.model.registry import ModelRecord, load_latest_model, save_model
from smaps.model.trainer import TrainedModel, train_model
from smaps.models import Direction


@pytest.fixture()
def conn(tmp_path: object) -> object:
    """In-memory DB with schema applied."""
    c = get_connection(":memory:")
    ensure_schema(c)
    return c


@pytest.fixture()
def models_dir(tmp_path):  # type: ignore[no-untyped-def]
    """Temporary directory for model artifacts."""
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture()
def trained_model() -> TrainedModel:
    """A simple trained model for testing."""
    rng = np.random.RandomState(42)
    n = 50
    features_df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
    })
    labels = pd.Series(rng.randint(0, 2, size=n))
    return train_model(features_df, labels)


class TestSaveModel:
    """Tests for save_model."""

    def test_saves_joblib_file(self, conn, models_dir, trained_model):  # type: ignore[no-untyped-def]
        record = save_model(conn, "AAPL", trained_model, models_dir)
        assert os.path.exists(record.artifact_path)

    def test_artifact_path_format(self, conn, models_dir, trained_model):  # type: ignore[no-untyped-def]
        record = save_model(conn, "AAPL", trained_model, models_dir)
        assert record.artifact_path.endswith("AAPL_v1.joblib")

    def test_returns_model_record(self, conn, models_dir, trained_model):  # type: ignore[no-untyped-def]
        record = save_model(conn, "AAPL", trained_model, models_dir)
        assert isinstance(record, ModelRecord)
        assert record.ticker == "AAPL"
        assert record.version == 1
        assert record.id is not None

    def test_metrics_json_stored(self, conn, models_dir, trained_model):  # type: ignore[no-untyped-def]
        record = save_model(conn, "AAPL", trained_model, models_dir)
        metrics = json.loads(record.metrics_json)
        assert "accuracy" in metrics

    def test_version_auto_increments(self, conn, models_dir, trained_model):  # type: ignore[no-untyped-def]
        r1 = save_model(conn, "AAPL", trained_model, models_dir)
        r2 = save_model(conn, "AAPL", trained_model, models_dir)
        assert r1.version == 1
        assert r2.version == 2
        assert r2.artifact_path.endswith("AAPL_v2.joblib")

    def test_different_tickers_independent_versions(self, conn, models_dir, trained_model):  # type: ignore[no-untyped-def]
        r_aapl = save_model(conn, "AAPL", trained_model, models_dir)
        r_msft = save_model(conn, "MSFT", trained_model, models_dir)
        assert r_aapl.version == 1
        assert r_msft.version == 1

    def test_creates_models_dir_if_missing(self, conn, tmp_path, trained_model):  # type: ignore[no-untyped-def]
        new_dir = tmp_path / "new_models"
        assert not new_dir.exists()
        record = save_model(conn, "AAPL", trained_model, new_dir)
        assert new_dir.exists()
        assert os.path.exists(record.artifact_path)


class TestLoadLatestModel:
    """Tests for load_latest_model."""

    def test_round_trip_save_load(self, conn, models_dir, trained_model):  # type: ignore[no-untyped-def]
        save_model(conn, "AAPL", trained_model, models_dir)
        result = load_latest_model(conn, "AAPL")
        assert result is not None
        loaded_model, record = result
        assert isinstance(loaded_model, TrainedModel)
        assert record.ticker == "AAPL"
        assert record.version == 1

    def test_loaded_model_produces_predictions(self, conn, models_dir, trained_model):  # type: ignore[no-untyped-def]
        save_model(conn, "AAPL", trained_model, models_dir)
        result = load_latest_model(conn, "AAPL")
        assert result is not None
        loaded_model, _ = result
        direction, confidence = loaded_model.predict({"f1": 0.5, "f2": -0.3})
        assert isinstance(direction, Direction)
        assert 0.0 <= confidence <= 1.0

    def test_loads_latest_version(self, conn, models_dir, trained_model):  # type: ignore[no-untyped-def]
        save_model(conn, "AAPL", trained_model, models_dir)
        save_model(conn, "AAPL", trained_model, models_dir)
        result = load_latest_model(conn, "AAPL")
        assert result is not None
        _, record = result
        assert record.version == 2

    def test_no_model_returns_none(self, conn):  # type: ignore[no-untyped-def]
        result = load_latest_model(conn, "AAPL")
        assert result is None

    def test_missing_artifact_returns_none(self, conn, models_dir, trained_model):  # type: ignore[no-untyped-def]
        record = save_model(conn, "AAPL", trained_model, models_dir)
        os.remove(record.artifact_path)
        result = load_latest_model(conn, "AAPL")
        assert result is None


class TestModelRegistry:
    """Tests for the model_registry table."""

    def test_registry_row_persisted(self, conn, models_dir, trained_model):  # type: ignore[no-untyped-def]
        save_model(conn, "AAPL", trained_model, models_dir)
        cur = conn.execute("SELECT COUNT(*) FROM model_registry WHERE ticker = 'AAPL'")
        assert cur.fetchone()[0] == 1

    def test_trained_at_is_iso_datetime(self, conn, models_dir, trained_model):  # type: ignore[no-untyped-def]
        save_model(conn, "AAPL", trained_model, models_dir)
        cur = conn.execute("SELECT trained_at FROM model_registry WHERE ticker = 'AAPL'")
        trained_at_str = cur.fetchone()[0]
        dt = datetime.datetime.fromisoformat(trained_at_str)
        assert dt.tzinfo is not None  # UTC timezone aware
