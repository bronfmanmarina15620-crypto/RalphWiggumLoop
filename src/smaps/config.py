"""Configuration loaded from environment variables with sensible defaults."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings. Override via SMAPS_* env vars."""

    tickers: list[str] = ["PLTR"]
    db_path: str = "data/smaps.sqlite"
    log_level: str = "INFO"

    twitter_enabled: bool = False
    twitter_api_key: str = ""
    twitter_api_secret: str = ""
    twitter_access_token: str = ""
    twitter_access_secret: str = ""

    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    model_config = {"env_prefix": "SMAPS_"}
