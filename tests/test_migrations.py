"""Tests for schema versioning and migrations."""

from __future__ import annotations

from smaps.db import (
    SCHEMA_VERSION,
    ensure_schema,
    get_connection,
    get_schema_version,
    migrate,
)


def test_fresh_db_creates_schema_migrations_and_sets_version():
    """Fresh DB -> ensure_schema creates schema_migrations at version 1."""
    conn = get_connection(":memory:")
    ensure_schema(conn)

    # schema_migrations table exists
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    )
    assert cur.fetchone() is not None

    # Version is SCHEMA_VERSION (1)
    assert get_schema_version(conn) == SCHEMA_VERSION
    conn.close()


def test_migrate_from_version_zero():
    """DB with schema_migrations at version 0 -> migrate brings it to 1."""
    conn = get_connection(":memory:")
    # Manually create the tracking table with no rows (version 0)
    conn.execute("CREATE TABLE schema_migrations (version INTEGER NOT NULL)")
    conn.commit()
    assert get_schema_version(conn) == 0

    # Run migrate — should apply migration_001
    migrate(conn)

    # ohlcv_daily should now exist
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv_daily'"
    )
    assert cur.fetchone() is not None
    conn.close()


def test_ensure_schema_idempotent():
    """Calling ensure_schema twice leaves version at 1 with no errors."""
    conn = get_connection(":memory:")
    ensure_schema(conn)
    ensure_schema(conn)

    assert get_schema_version(conn) == SCHEMA_VERSION

    # ohlcv_daily still has the right columns
    cur = conn.execute("PRAGMA table_info(ohlcv_daily)")
    columns = {row[1] for row in cur.fetchall()}
    expected = {"ticker", "date", "open", "high", "low", "close", "adj_close", "volume"}
    assert expected == columns
    conn.close()
