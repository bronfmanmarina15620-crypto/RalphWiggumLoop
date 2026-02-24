"""RSS-based sentiment collector using Google News headlines."""

from __future__ import annotations

import datetime
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
from urllib.parse import quote

from smaps.models import SentimentScore

_SOURCE = "google_news_rss"

# Simple keyword lists for headline sentiment heuristic
_POSITIVE = frozenset(
    {
        "surge",
        "surges",
        "jump",
        "jumps",
        "rally",
        "rallies",
        "gain",
        "gains",
        "rise",
        "rises",
        "soar",
        "soars",
        "record",
        "upgrade",
        "upgrades",
        "buy",
        "bullish",
        "beat",
        "beats",
        "strong",
        "profit",
        "profits",
        "growth",
        "boost",
        "positive",
        "outperform",
        "up",
        "high",
        "recover",
        "recovers",
    }
)

_NEGATIVE = frozenset(
    {
        "drop",
        "drops",
        "fall",
        "falls",
        "plunge",
        "plunges",
        "crash",
        "decline",
        "declines",
        "loss",
        "losses",
        "sell",
        "bearish",
        "miss",
        "misses",
        "weak",
        "downgrade",
        "downgrades",
        "risk",
        "fear",
        "fears",
        "down",
        "low",
        "cut",
        "cuts",
        "slump",
        "warning",
        "negative",
        "underperform",
    }
)


def _score_headline(headline: str) -> float:
    """Score a single headline using keyword matching.

    Returns a value between -1.0 and 1.0.
    """
    words = set(headline.lower().split())
    pos = len(words & _POSITIVE)
    neg = len(words & _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / total))


def _fetch_rss(ticker: str) -> list[str]:
    """Fetch headline titles from Google News RSS for a ticker."""
    url = (
        f"https://news.google.com/rss/search?q={quote(ticker)}+stock&hl=en-US&gl=US&ceid=US:en"
    )
    req = Request(url, headers={"User-Agent": "smaps/0.1"})
    with urlopen(req, timeout=10) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    return [item.text for item in root.iter("title") if item.text]


def fetch_sentiment(ticker: str, date: datetime.date) -> SentimentScore:
    """Fetch a daily sentiment score for a ticker.

    Fetches recent news headlines via Google News RSS and computes
    a simple keyword-based sentiment score.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        date: The date to associate the sentiment with.

    Returns:
        SentimentScore with averaged headline sentiment.
    """
    headlines = _fetch_rss(ticker)
    if not headlines:
        return SentimentScore(
            ticker=ticker, date=date, score=0.0, source=_SOURCE
        )

    scores = [_score_headline(h) for h in headlines]
    avg = sum(scores) / len(scores)
    # Clamp to [-1, 1] range
    avg = max(-1.0, min(1.0, avg))

    return SentimentScore(
        ticker=ticker, date=date, score=round(avg, 4), source=_SOURCE
    )
