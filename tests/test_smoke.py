"""Smoke tests: verify the package is importable and DB schema works."""

from __future__ import annotations


def test_import_smaps():
    """Package can be imported and exposes a version."""
    import smaps

    assert hasattr(smaps, "__version__")
    assert isinstance(smaps.__version__, str)


def test_config_defaults():
    """Settings loads with all defaults (no env vars required)."""
    from smaps.config import Settings

    s = Settings()
    assert s.db_path == "data/smaps.sqlite"
    assert s.log_level == "INFO"
    assert "PLTR" in s.tickers


def test_get_logger():
    """get_logger returns a usable logger."""
    from smaps.logging import get_logger

    logger = get_logger("test_smoke")
    assert logger.name == "test_smoke"

    logger_with_run = get_logger("test_run", run_id="abc123")
    assert logger_with_run.name == "test_run"


def test_ensure_schema():
    """ensure_schema creates ohlcv_daily table in an in-memory DB."""
    from smaps.db import ensure_schema, get_connection

    conn = get_connection(":memory:")
    ensure_schema(conn)

    # Verify table exists
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv_daily'"
    )
    assert cur.fetchone() is not None

    # Verify idempotent: calling again does not raise
    ensure_schema(conn)
    conn.close()


def test_ensure_schema_columns():
    """ohlcv_daily has the expected columns and primary key."""
    from smaps.db import ensure_schema, get_connection

    conn = get_connection(":memory:")
    ensure_schema(conn)

    cur = conn.execute("PRAGMA table_info(ohlcv_daily)")
    columns = {row[1] for row in cur.fetchall()}
    expected = {"ticker", "date", "open", "high", "low", "close", "adj_close", "volume"}
    assert expected == columns

    conn.close()
