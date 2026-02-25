"""Feature pipeline protocol for SMAPS.

All feature pipelines must implement the ``FeaturePipeline`` protocol so that
they can be composed and swapped transparently.
"""

from __future__ import annotations

import datetime
from typing import Protocol


class FeaturePipeline(Protocol):
    """Interface that every feature pipeline must satisfy.

    Implementations compute a feature vector for a given ticker as of a
    specific date.

    **No-future-data contract:** Implementations MUST only use data with
    dates **on or before** ``as_of_date``.  Any data dated after
    ``as_of_date`` must be excluded to prevent look-ahead bias.  This
    invariant is critical for the integrity of back-tests and live
    predictions alike.
    """

    def transform(
        self, ticker: str, as_of_date: datetime.date
    ) -> dict[str, float]:
        """Return a feature vector for *ticker* as of *as_of_date*.

        Args:
            ticker: The stock ticker symbol (e.g. ``"AAPL"``).
            as_of_date: The reference date.  Only data dated
                ``<= as_of_date`` may be used.

        Returns:
            A mapping of feature names to their float values.
        """
        ...  # pragma: no cover
