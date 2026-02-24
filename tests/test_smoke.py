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


def test_config_env_overrides(monkeypatch):
    """Env vars with SMAPS_ prefix override default settings."""
    from smaps.config import Settings

    monkeypatch.setenv("SMAPS_DB_PATH", "/tmp/test.db")
    monkeypatch.setenv("SMAPS_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("SMAPS_TICKERS", '["GOOG","TSLA"]')

    s = Settings()
    assert s.db_path == "/tmp/test.db"
    assert s.log_level == "DEBUG"
    assert s.tickers == ["GOOG", "TSLA"]


def test_get_logger():
    """get_logger returns a logging.Logger with correct name."""
    import logging

    from smaps.logging import get_logger

    logger = get_logger("test_smoke")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_smoke"

    logger_with_run = get_logger("test_run", run_id="abc123")
    assert isinstance(logger_with_run, logging.Logger)
    assert logger_with_run.name == "test_run"


def test_get_logger_output_format(capfd):
    """Logger output includes ISO timestamp, level, name, and run_id."""
    import logging

    from smaps.logging import get_logger

    # Use unique names to avoid handler reuse from previous tests
    logger = get_logger("fmt_test_run", run_id="RUN42")
    logger.setLevel(logging.INFO)
    logger.info("hello")
    captured = capfd.readouterr()
    # ISO timestamp like 2025-01-15T10:30:00
    assert "T" in captured.err  # ISO date separator
    assert "INFO" in captured.err
    assert "fmt_test_run" in captured.err
    assert "RUN42" in captured.err
    assert "hello" in captured.err


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


def test_schema_migrations_tracks_version():
    """schema_migrations table tracks applied version after ensure_schema."""
    from smaps.db import SCHEMA_VERSION, ensure_schema, get_connection

    conn = get_connection(":memory:")
    ensure_schema(conn)

    # Verify schema_migrations table exists
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    )
    assert cur.fetchone() is not None

    # Verify tracked version matches SCHEMA_VERSION
    cur = conn.execute("SELECT version FROM schema_migrations")
    row = cur.fetchone()
    assert row is not None
    assert row[0] == SCHEMA_VERSION

    # Verify only one version row exists (not accumulated)
    cur = conn.execute("SELECT COUNT(*) FROM schema_migrations")
    assert cur.fetchone()[0] == 1

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
