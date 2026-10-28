"""News collection for the news-sentiment research track."""
from datetime import datetime

# First collector run. Articles published before it are backlog and excluded
# from the study. This is the single source of the cutoff: labeling and later
# aggregation must import it from here.
COLLECTION_START = "2026-10-24T22:01:53Z"


def in_study_period(published_utc, start=COLLECTION_START):
    """Return True if an article was published at or after ``start``.

    Parameters
    ----------
    published_utc : str
        ISO 8601 UTC time, for example ``"2026-10-25T08:00:00Z"``.
    start : str
        ISO 8601 UTC cutoff (inclusive). Defaults to ``COLLECTION_START``.

    Raises
    ------
    ValueError
        If either time cannot be parsed or has no timezone.
    """
    published, cutoff = datetime.fromisoformat(published_utc), datetime.fromisoformat(start)
    if published.tzinfo is None or cutoff.tzinfo is None:
        raise ValueError("times must include a timezone")
    return published >= cutoff
