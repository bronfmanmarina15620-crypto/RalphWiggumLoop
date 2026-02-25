"""Twitter/X and Telegram notifications for pipeline run status."""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

import tweepy

from smaps.config import Settings

logger = logging.getLogger(__name__)


def format_status_message(result: dict[str, Any]) -> str:
    """Build a concise tweet from a pipeline result dict.

    Example output::

        SMAPS Daily Run 2026-02-25
        PLTR: UP (87%) | Retrain: not needed
        Elapsed: 12.3s
    """
    lines: list[str] = [f"SMAPS Daily Run {result['date']}"]

    for ticker, info in result.get("tickers", {}).items():
        direction = info.get("direction", "?")
        confidence = info.get("confidence")
        conf_str = f" ({confidence:.0%})" if confidence is not None else ""
        retrain = info.get("retrain", "?")
        status = "error" if info.get("predict") == "error" else f"{direction}{conf_str}"
        lines.append(f"{ticker}: {status} | Retrain: {retrain}")

    elapsed = result.get("elapsed", 0)
    lines.append(f"Elapsed: {elapsed:.1f}s")

    return "\n".join(lines)


def send_twitter_update(result: dict[str, Any], settings: Settings) -> None:
    """Post a status tweet summarising the pipeline run.

    Does nothing when *twitter_enabled* is False or credentials are missing.
    Exceptions are logged but never propagated — notification failure must
    not break the pipeline.
    """
    if not settings.twitter_enabled:
        return

    if not all([
        settings.twitter_api_key,
        settings.twitter_api_secret,
        settings.twitter_access_token,
        settings.twitter_access_secret,
    ]):
        logger.warning("twitter_enabled=true but credentials are missing — skipping tweet")
        return

    message = format_status_message(result)

    try:
        client = tweepy.Client(
            consumer_key=settings.twitter_api_key,
            consumer_secret=settings.twitter_api_secret,
            access_token=settings.twitter_access_token,
            access_token_secret=settings.twitter_access_secret,
        )
        client.create_tweet(text=message)
        logger.info("tweet_sent length=%d", len(message))
    except Exception:
        logger.exception("tweet_failed")


def send_telegram_update(result: dict[str, Any], settings: Settings) -> None:
    """Send a Telegram message summarising the pipeline run.

    Does nothing when *telegram_enabled* is False or credentials are missing.
    Exceptions are logged but never propagated.
    """
    if not settings.telegram_enabled:
        return

    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.warning("telegram_enabled=true but bot_token/chat_id missing — skipping")
        return

    message = format_status_message(result)

    try:
        url = (
            f"https://api.telegram.org/bot{settings.telegram_bot_token}"
            f"/sendMessage"
        )
        payload = json.dumps({
            "chat_id": settings.telegram_chat_id,
            "text": message,
        }).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        logger.info("telegram_sent length=%d", len(message))
    except Exception:
        logger.exception("telegram_failed")
