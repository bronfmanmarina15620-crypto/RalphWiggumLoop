"""Tests for RSS-based sentiment collector."""

from __future__ import annotations

import datetime
from unittest.mock import patch

from smaps.collectors.sentiment import (
    _score_headline,
    fetch_sentiment,
)
from smaps.models import SentimentScore


class TestScoreHeadline:
    """Tests for the keyword-based headline scoring function."""

    def test_positive_headline(self) -> None:
        """Headline with positive keywords gets positive score."""
        score = _score_headline("AAPL stock surges on strong earnings beat")
        assert score > 0.0

    def test_negative_headline(self) -> None:
        """Headline with negative keywords gets negative score."""
        score = _score_headline("AAPL shares drop on weak sales and downgrade")
        assert score < 0.0

    def test_neutral_headline(self) -> None:
        """Headline with no sentiment keywords gets zero score."""
        score = _score_headline("Apple announces new product launch")
        assert score == 0.0

    def test_mixed_headline(self) -> None:
        """Headline with equal positive and negative keywords gets zero."""
        score = _score_headline("stock gains despite fear")
        assert score == 0.0

    def test_score_range(self) -> None:
        """Score is always clamped between -1 and 1."""
        score = _score_headline("surge rally gain rise jump soar boost record")
        assert -1.0 <= score <= 1.0


class TestFetchSentiment:
    """Tests for fetch_sentiment with mocked RSS feed."""

    _MOCK_RSS = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>AAPL stock - Google News</title>
        <item><title>AAPL surges on strong earnings</title></item>
        <item><title>Apple stock gains after upgrade</title></item>
        <item><title>Tech sector rally boosts AAPL</title></item>
      </channel>
    </rss>"""

    _MOCK_RSS_NEGATIVE = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>AAPL stock - Google News</title>
        <item><title>AAPL drops on weak guidance</title></item>
        <item><title>Analyst downgrade sends stock down</title></item>
      </channel>
    </rss>"""

    _MOCK_RSS_EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>No results - Google News</title>
      </channel>
    </rss>"""

    @patch("smaps.collectors.sentiment.urlopen")
    def test_returns_sentiment_score(self, mock_urlopen: object) -> None:
        """fetch_sentiment returns a SentimentScore object."""
        mock_urlopen.return_value.__enter__ = lambda s: s  # type: ignore[union-attr]
        mock_urlopen.return_value.__exit__ = lambda s, *a: None  # type: ignore[union-attr]
        mock_urlopen.return_value.read.return_value = self._MOCK_RSS.encode()  # type: ignore[union-attr]

        result = fetch_sentiment("AAPL", datetime.date(2025, 1, 15))

        assert isinstance(result, SentimentScore)

    @patch("smaps.collectors.sentiment.urlopen")
    def test_positive_sentiment(self, mock_urlopen: object) -> None:
        """Positive headlines produce a positive sentiment score."""
        mock_urlopen.return_value.__enter__ = lambda s: s  # type: ignore[union-attr]
        mock_urlopen.return_value.__exit__ = lambda s, *a: None  # type: ignore[union-attr]
        mock_urlopen.return_value.read.return_value = self._MOCK_RSS.encode()  # type: ignore[union-attr]

        result = fetch_sentiment("AAPL", datetime.date(2025, 1, 15))

        assert result.score > 0.0
        assert result.ticker == "AAPL"
        assert result.date == datetime.date(2025, 1, 15)
        assert result.source == "google_news_rss"

    @patch("smaps.collectors.sentiment.urlopen")
    def test_negative_sentiment(self, mock_urlopen: object) -> None:
        """Negative headlines produce a negative sentiment score."""
        mock_urlopen.return_value.__enter__ = lambda s: s  # type: ignore[union-attr]
        mock_urlopen.return_value.__exit__ = lambda s, *a: None  # type: ignore[union-attr]
        mock_urlopen.return_value.read.return_value = self._MOCK_RSS_NEGATIVE.encode()  # type: ignore[union-attr]

        result = fetch_sentiment("AAPL", datetime.date(2025, 1, 15))

        assert result.score < 0.0

    @patch("smaps.collectors.sentiment.urlopen")
    def test_no_headlines_returns_zero(self, mock_urlopen: object) -> None:
        """Empty RSS feed returns score of 0.0."""
        mock_urlopen.return_value.__enter__ = lambda s: s  # type: ignore[union-attr]
        mock_urlopen.return_value.__exit__ = lambda s, *a: None  # type: ignore[union-attr]
        mock_urlopen.return_value.read.return_value = self._MOCK_RSS_EMPTY.encode()  # type: ignore[union-attr]

        result = fetch_sentiment("AAPL", datetime.date(2025, 1, 15))

        assert result.score == 0.0

    @patch("smaps.collectors.sentiment.urlopen")
    def test_score_in_valid_range(self, mock_urlopen: object) -> None:
        """Returned score is always between -1 and 1."""
        mock_urlopen.return_value.__enter__ = lambda s: s  # type: ignore[union-attr]
        mock_urlopen.return_value.__exit__ = lambda s, *a: None  # type: ignore[union-attr]
        mock_urlopen.return_value.read.return_value = self._MOCK_RSS.encode()  # type: ignore[union-attr]

        result = fetch_sentiment("AAPL", datetime.date(2025, 1, 15))

        assert -1.0 <= result.score <= 1.0
