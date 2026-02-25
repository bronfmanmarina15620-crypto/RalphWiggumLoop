"""Tests for US-603: CLI entry point (--tickers, --date, --dry-run, --help)."""

from __future__ import annotations

import datetime
import logging
from unittest.mock import patch

import pytest

from smaps.pipeline import _build_parser, _run_dry, main


class TestBuildParser:
    """Verify argparse configuration."""

    def test_help_flag_exits_zero(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_tickers_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--tickers", "AAPL,MSFT"])
        assert args.tickers == "AAPL,MSFT"

    def test_date_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--date", "2025-01-15"])
        assert args.date == "2025-01-15"

    def test_dry_run_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.tickers is None
        assert args.date is None
        assert args.dry_run is False


class TestMain:
    """Verify the main() entry point."""

    @patch("smaps.pipeline.run_pipeline")
    @patch("smaps.config.Settings")
    def test_tickers_and_date_args(
        self, mock_settings_cls: object, mock_run: object
    ) -> None:
        """--tickers AAPL,MSFT --date 2025-01-15 passes correct args."""
        mock_settings_cls.return_value.tickers = ["PLTR"]  # type: ignore[union-attr]
        mock_settings_cls.return_value.db_path = ":memory:"  # type: ignore[union-attr]

        main(["--tickers", "AAPL,MSFT", "--date", "2025-01-15"])

        mock_run.assert_called_once_with(  # type: ignore[union-attr]
            tickers=["AAPL", "MSFT"],
            date=datetime.date(2025, 1, 15),
            db_path=":memory:",
        )

    @patch("smaps.pipeline.run_pipeline")
    @patch("smaps.config.Settings")
    def test_tickers_from_args(
        self, mock_settings_cls: object, mock_run: object
    ) -> None:
        mock_settings_cls.return_value.tickers = ["PLTR"]  # type: ignore[union-attr]
        mock_settings_cls.return_value.db_path = ":memory:"  # type: ignore[union-attr]

        main(["--tickers", "AAPL,MSFT", "--date", "2025-01-15"])

        mock_run.assert_called_once_with(  # type: ignore[union-attr]
            tickers=["AAPL", "MSFT"],
            date=datetime.date(2025, 1, 15),
            db_path=":memory:",
        )

    @patch("smaps.pipeline.run_pipeline")
    @patch("smaps.config.Settings")
    def test_defaults_from_settings(
        self, mock_settings_cls: object, mock_run: object
    ) -> None:
        """No args → tickers/db_path from Settings, date=today."""
        mock_settings_cls.return_value.tickers = ["PLTR"]  # type: ignore[union-attr]
        mock_settings_cls.return_value.db_path = "data/smaps.sqlite"  # type: ignore[union-attr]

        main([])

        mock_run.assert_called_once()  # type: ignore[union-attr]
        kwargs = mock_run.call_args.kwargs  # type: ignore[union-attr]
        assert kwargs["tickers"] == ["PLTR"]
        assert kwargs["db_path"] == "data/smaps.sqlite"
        assert kwargs["date"] == datetime.date.today()

    @patch("smaps.pipeline.run_pipeline")
    @patch("smaps.config.Settings")
    def test_dry_run_does_not_call_run_pipeline(
        self, mock_settings_cls: object, mock_run: object
    ) -> None:
        """--dry-run logs steps but does NOT call run_pipeline."""
        mock_settings_cls.return_value.tickers = ["AAPL"]  # type: ignore[union-attr]
        mock_settings_cls.return_value.db_path = ":memory:"  # type: ignore[union-attr]

        main(["--dry-run"])

        mock_run.assert_not_called()  # type: ignore[union-attr]

    @patch("smaps.pipeline.run_pipeline")
    @patch("smaps.config.Settings")
    def test_tickers_whitespace_stripped(
        self, mock_settings_cls: object, mock_run: object
    ) -> None:
        """Spaces around ticker names are stripped."""
        mock_settings_cls.return_value.tickers = ["PLTR"]  # type: ignore[union-attr]
        mock_settings_cls.return_value.db_path = ":memory:"  # type: ignore[union-attr]

        main(["--tickers", " AAPL , MSFT ", "--date", "2025-01-15"])

        kwargs = mock_run.call_args.kwargs  # type: ignore[union-attr]
        assert kwargs["tickers"] == ["AAPL", "MSFT"]


class TestDryRun:
    """Verify dry-run logging."""

    def test_logs_all_steps_per_ticker(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="smaps.pipeline"):
            _run_dry(["AAPL", "MSFT"], datetime.date(2025, 1, 15))

        messages = caplog.text
        for ticker in ("AAPL", "MSFT"):
            for step in ("ingest", "predict", "evaluate", "retrain"):
                assert f"step={step}" in messages
                assert ticker in messages

    def test_logs_dry_run_complete(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="smaps.pipeline"):
            _run_dry(["AAPL"], datetime.date(2025, 1, 15))

        assert "dry_run complete" in caplog.text

    def test_logs_tickers_and_date(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="smaps.pipeline"):
            _run_dry(["AAPL", "GOOG"], datetime.date(2025, 3, 20))

        assert "AAPL,GOOG" in caplog.text
        assert "2025-03-20" in caplog.text
