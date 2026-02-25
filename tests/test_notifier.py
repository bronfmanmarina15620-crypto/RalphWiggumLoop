"""Tests for smaps.notifier — message formatting, tweet and Telegram dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from smaps.config import Settings
from smaps.notifier import format_status_message, send_telegram_update, send_twitter_update


SAMPLE_RESULT: dict = {
    "run_id": "abc12345",
    "date": "2026-02-25",
    "tickers": {
        "PLTR": {
            "ingest": "ok",
            "predict": "ok",
            "direction": "UP",
            "confidence": 0.87,
            "retrain": "not_needed",
        },
    },
    "elapsed": 12.34,
}


def test_format_single_ticker() -> None:
    msg = format_status_message(SAMPLE_RESULT)
    assert "SMAPS Daily Run 2026-02-25" in msg
    assert "PLTR: UP (87%)" in msg
    assert "Retrain: not_needed" in msg
    assert "12.3s" in msg


def test_format_multiple_tickers() -> None:
    result = {
        "date": "2026-02-25",
        "tickers": {
            "PLTR": {"predict": "ok", "direction": "UP", "confidence": 0.87, "retrain": "not_needed"},
            "AAPL": {"predict": "ok", "direction": "DOWN", "confidence": 0.65, "retrain": "retrained"},
        },
        "elapsed": 25.0,
    }
    msg = format_status_message(result)
    assert "PLTR: UP (87%)" in msg
    assert "AAPL: DOWN (65%)" in msg
    assert "Retrain: retrained" in msg


def test_format_prediction_error() -> None:
    result = {
        "date": "2026-02-25",
        "tickers": {
            "PLTR": {"predict": "error", "retrain": "not_needed"},
        },
        "elapsed": 1.0,
    }
    msg = format_status_message(result)
    assert "PLTR: error" in msg


def test_send_skipped_when_disabled() -> None:
    settings = Settings(twitter_enabled=False)
    with patch("smaps.notifier.tweepy") as mock_tweepy:
        send_twitter_update(SAMPLE_RESULT, settings)
        mock_tweepy.Client.assert_not_called()


def test_send_skipped_when_credentials_missing() -> None:
    settings = Settings(twitter_enabled=True, twitter_api_key="key", twitter_api_secret="")
    with patch("smaps.notifier.tweepy") as mock_tweepy:
        send_twitter_update(SAMPLE_RESULT, settings)
        mock_tweepy.Client.assert_not_called()


@patch("smaps.notifier.tweepy")
def test_send_posts_tweet(mock_tweepy: MagicMock) -> None:
    settings = Settings(
        twitter_enabled=True,
        twitter_api_key="key",
        twitter_api_secret="secret",
        twitter_access_token="token",
        twitter_access_secret="tsecret",
    )
    mock_client = MagicMock()
    mock_tweepy.Client.return_value = mock_client

    send_twitter_update(SAMPLE_RESULT, settings)

    mock_tweepy.Client.assert_called_once_with(
        consumer_key="key",
        consumer_secret="secret",
        access_token="token",
        access_token_secret="tsecret",
    )
    mock_client.create_tweet.assert_called_once()
    tweet_text = mock_client.create_tweet.call_args.kwargs["text"]
    assert "SMAPS Daily Run 2026-02-25" in tweet_text
    assert "PLTR" in tweet_text


@patch("smaps.notifier.tweepy")
def test_send_does_not_raise_on_api_error(mock_tweepy: MagicMock) -> None:
    settings = Settings(
        twitter_enabled=True,
        twitter_api_key="key",
        twitter_api_secret="secret",
        twitter_access_token="token",
        twitter_access_secret="tsecret",
    )
    mock_tweepy.Client.return_value.create_tweet.side_effect = RuntimeError("API down")

    # Should not raise
    send_twitter_update(SAMPLE_RESULT, settings)


# --- Telegram tests ---


def test_telegram_skipped_when_disabled() -> None:
    settings = Settings(telegram_enabled=False)
    with patch("smaps.notifier.urllib.request.urlopen") as mock_urlopen:
        send_telegram_update(SAMPLE_RESULT, settings)
        mock_urlopen.assert_not_called()


def test_telegram_skipped_when_credentials_missing() -> None:
    settings = Settings(telegram_enabled=True, telegram_bot_token="tok", telegram_chat_id="")
    with patch("smaps.notifier.urllib.request.urlopen") as mock_urlopen:
        send_telegram_update(SAMPLE_RESULT, settings)
        mock_urlopen.assert_not_called()


@patch("smaps.notifier.urllib.request.urlopen")
def test_telegram_sends_message(mock_urlopen: MagicMock) -> None:
    settings = Settings(
        telegram_enabled=True,
        telegram_bot_token="123:ABC",
        telegram_chat_id="-100999",
    )

    send_telegram_update(SAMPLE_RESULT, settings)

    mock_urlopen.assert_called_once()
    req = mock_urlopen.call_args[0][0]
    assert "123:ABC" in req.full_url
    assert req.get_header("Content-type") == "application/json"

    import json
    body = json.loads(req.data)
    assert body["chat_id"] == "-100999"
    assert "SMAPS Daily Run 2026-02-25" in body["text"]
    assert "PLTR" in body["text"]


@patch("smaps.notifier.urllib.request.urlopen")
def test_telegram_does_not_raise_on_error(mock_urlopen: MagicMock) -> None:
    settings = Settings(
        telegram_enabled=True,
        telegram_bot_token="123:ABC",
        telegram_chat_id="-100999",
    )
    mock_urlopen.side_effect = RuntimeError("network error")

    # Should not raise
    send_telegram_update(SAMPLE_RESULT, settings)
