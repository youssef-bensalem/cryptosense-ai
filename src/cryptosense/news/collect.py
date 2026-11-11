"""Collect crypto news articles from RSS feeds into ``data/news/``.

Usage: ``python -m cryptosense.news.collect [--data-dir PATH]``

Articles are appended to ``articles.csv`` and one row per feed per run is
appended to ``runs.csv``. Existing rows are never rewritten. Articles without
a publication time that states its timezone are skipped, never guessed.
"""
import argparse
import csv
import hashlib
import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import requests

from cryptosense.news.feeds import FEEDS

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
USER_AGENT = "CryptoSenseAI-NewsCollector/0.1 (personal research; RSS reader)"
TIMEOUT_S = 20

ARTICLE_COLUMNS = ["id", "source", "title", "summary", "url", "published_utc", "fetched_utc"]
RUN_COLUMNS = ["run_utc", "source", "status", "new", "duplicate", "skipped", "error"]
TRACKING_PARAMS = {"ref", "source"}
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def normalize_url(url):
    """Normalize an article URL so that equivalent links compare equal.

    Lowercases the scheme and host, drops the fragment, removes tracking
    query parameters (``utm_*``, ``ref``, ``source``) and removes a trailing
    slash from the path. The order of the remaining parameters is kept.

    Parameters
    ----------
    url : str
        The article link as found in the feed.

    Returns
    -------
    str
        The normalized URL.
    """
    parts = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMS
    ]
    path = parts.path[:-1] if parts.path.endswith("/") else parts.path
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def article_id(url):
    """Return the article id: the SHA-256 hex digest of the normalized URL."""
    return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()


class _TextExtractor(HTMLParser):
    """Collect the text content of an HTML fragment, discarding tags."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def strip_html(text):
    """Remove HTML tags from ``text``, decode entities and collapse whitespace."""
    extractor = _TextExtractor()
    extractor.feed(text or "")
    extractor.close()
    return re.sub(r"\s+", " ", "".join(extractor.parts)).strip()


def parse_published_utc(raw):
    """Parse a feed publication time into a timezone-aware UTC datetime.

    Accepts RFC 822 dates (the RSS format, e.g. ``"Sat, 24 Oct 2026 18:00:00
    +0200"``) and ISO 8601 dates (the Atom format). The time must state its
    timezone; a time without one (or with RFC 822's "unknown" ``-0000``) is
    rejected rather than assumed to be UTC.

    Parameters
    ----------
    raw : str or None
        The publication time string from the feed entry.

    Returns
    -------
    datetime or None
        The time converted to UTC, or None if it is missing or unparseable.
    """
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    parsed = None
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def format_utc(moment):
    """Format a timezone-aware datetime as ISO 8601 UTC, e.g. ``2026-10-24T16:00:00Z``."""
    return moment.astimezone(timezone.utc).strftime(TIMESTAMP_FORMAT)


def fetch_feed(url):
    """Download a feed and return its raw bytes.

    Uses a descriptive User-Agent and a 20-second timeout.

    Raises
    ------
    requests.RequestException
        On network errors and on HTTP error statuses.
    """
    response = requests.get(url, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.content


class FeedError(Exception):
    """Raised when downloaded content is not a well-formed RSS or Atom feed."""


def parse_feed(content):
    """Parse raw feed bytes with feedparser.

    Raises
    ------
    FeedError
        If the XML is malformed or the content is not a recognized feed.
        A declared-encoding mismatch alone is tolerated.
    """
    parsed = feedparser.parse(content)
    if parsed.bozo and not isinstance(parsed.bozo_exception, feedparser.CharacterEncodingOverride):
        raise FeedError(f"malformed feed: {parsed.bozo_exception}")
    if not parsed.version:
        raise FeedError("content is not a recognized RSS or Atom feed")
    return parsed


def _read_ids(path):
    """Return the set of article ids already stored in ``path``."""
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["id"] for row in csv.DictReader(handle)}


def _append_rows(path, columns, rows):
    """Append ``rows`` to a CSV file, writing the header if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def collect(feeds=FEEDS, data_dir=DEFAULT_DATA_DIR, fetch=fetch_feed, now=None):
    """Fetch every feed once and append new articles and a run log to disk.

    Feeds are processed in order. An article is new if its id is in neither
    ``articles.csv`` nor an earlier feed of this run; otherwise it is counted
    as a duplicate and the first-seen row is kept. Entries without a link or
    without a parseable publication time are skipped and logged. A feed that
    fails (network error, HTTP error, malformed XML) is logged as failed and
    the remaining feeds are still collected.

    Parameters
    ----------
    feeds : list of dict
        Feeds with ``"source"`` and ``"url"`` keys.
    data_dir : Path
        The project data directory; files are written to ``data_dir / "news"``.
    fetch : callable
        ``fetch(url) -> bytes``. Inject a fake in tests to avoid the network.
    now : callable or None
        ``now() -> datetime`` (timezone-aware) for run and fetch times.
        Defaults to the current UTC time.

    Returns
    -------
    list of dict
        One run-log row per feed, as written to ``runs.csv``.
    """
    now = now or (lambda: datetime.now(timezone.utc))
    news_dir = Path(data_dir) / "news"
    articles_path = news_dir / "articles.csv"
    seen = _read_ids(articles_path)
    run_utc = format_utc(now())
    run_rows = []

    for feed in feeds:
        source = feed["source"]
        run_row = {"run_utc": run_utc, "source": source, "status": "ok",
                   "new": 0, "duplicate": 0, "skipped": 0, "error": ""}
        try:
            parsed = parse_feed(fetch(feed["url"]))
        except Exception as exc:
            logger.warning("feed %s failed: %s", source, exc)
            run_row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            run_rows.append(run_row)
            continue

        fetched_utc = format_utc(now())
        new_rows = []
        for entry in parsed.entries:
            link = entry.get("link")
            published = parse_published_utc(entry.get("published"))
            if not link or published is None:
                reason = "no link" if not link else f"unparseable publication time {entry.get('published')!r}"
                logger.info("skipped %s entry %r: %s", source, entry.get("title", ""), reason)
                run_row["skipped"] += 1
                continue
            entry_id = article_id(link)
            if entry_id in seen:
                run_row["duplicate"] += 1
                continue
            seen.add(entry_id)
            new_rows.append({
                "id": entry_id,
                "source": source,
                "title": (entry.get("title") or "").strip(),
                "summary": strip_html(entry.get("summary")),
                "url": normalize_url(link),
                "published_utc": format_utc(published),
                "fetched_utc": fetched_utc,
            })
        _append_rows(articles_path, ARTICLE_COLUMNS, new_rows)
        run_row["new"] = len(new_rows)
        run_rows.append(run_row)

    _append_rows(news_dir / "runs.csv", RUN_COLUMNS, run_rows)
    return run_rows


def format_summary(run_rows):
    """Format per-feed and total new / duplicate / skipped / failed counts as a table."""
    lines = [f"{'source':<18}{'new':>6}{'duplicate':>11}{'skipped':>9}{'failed':>8}"]
    totals = {"new": 0, "duplicate": 0, "skipped": 0, "failed": 0}
    for row in run_rows:
        failed = int(row["status"] == "failed")
        lines.append(f"{row['source']:<18}{row['new']:>6}{row['duplicate']:>11}"
                     f"{row['skipped']:>9}{failed:>8}")
        for key in ("new", "duplicate", "skipped"):
            totals[key] += row[key]
        totals["failed"] += failed
    lines.append(f"{'total':<18}{totals['new']:>6}{totals['duplicate']:>11}"
                 f"{totals['skipped']:>9}{totals['failed']:>8}")
    return "\n".join(lines)


def main(argv=None):
    """Command-line entry point: collect all feeds and print the counts."""
    parser = argparse.ArgumentParser(description="Collect crypto news RSS feeds.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="project data directory (default: %(default)s)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(format_summary(collect(data_dir=args.data_dir)))


if __name__ == "__main__":
    main()
