"""Configuration loaded from environment variables with sensible defaults."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings. Override via SMAPS_* env vars."""

    tickers: list[str] = ["PLTR"]
    db_path: str = "data/smaps.sqlite"
    log_level: str = "INFO"

    model_config = {"env_prefix": "SMAPS_"}
