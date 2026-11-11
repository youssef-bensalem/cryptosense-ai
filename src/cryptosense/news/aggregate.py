"""Aggregate labeled news articles into daily sentiment scores and coverage.

Usage: ``python -m cryptosense.news.aggregate [--data-dir PATH]``

Implements sections 4-6 of PREREGISTRATION_NEWS.md. Inputs are
``articles.csv``, ``labels.csv`` and ``runs.csv`` in ``data/news/``; the
output is ``daily_sentiment.csv`` in the same directory, rewritten on every
run. Input files are only read. This module never loads or uses price data.
The CLI prints coverage figures only, never a sentiment value.

Article rules (section 4): an article counts only if it was published at or
after ``COLLECTION_START``, was fetched at most 24 hours after publication,
has a label with ``model_reported`` ``EXPECTED_REPORTED_MODEL`` and
``prompt_version`` ``PROMPT_VERSION`` (the first such row in ``labels.csv``
if there are several), and is not a decrypt newsletter (a decrypt title with
two or more "!"). Its day is the UTC date of ``published_utc``. The rules are
applied before duplicates are grouped, so an article that fails a rule is
never a group member.

Implementation clarifications made before any results (2026-10-28):

1. Duplicates (rule 6): titles identical after lowercasing and removing
   punctuation, published within 48 hours, form one group. The group belongs
   to the day of its earliest member, and its sentiment is the mean over the
   members published on that same UTC day only; later members are dropped.
2. Coverage (section 6): an hour counts if at least one collector run started
   in it and every source in that run had status ok.
3. Group relevance: a group item is relevant for an asset (market or BTC,
   separately) if at least one of its same-day members is relevant for that
   asset; its sentiment for that asset is the mean over the same-day members
   relevant for that asset.
4. Duplicate chains: in order of publication time, an article joins the open
   group with its normalized title if it was published within 48 hours of
   that group's earliest member; otherwise it starts a new group.
5. Title normalization and confidence weighting: removing punctuation means
   removing every Unicode punctuation character (category P, including curly
   quotes and dashes) except "%", which is kept although Unicode classes it
   as punctuation; all symbols (categories S*, including "$") are kept. Then
   runs of whitespace are collapsed and the result is trimmed. With ``weight_by_confidence``, each item is
   weighted by its confidence; a group's confidence for an asset is the mean
   confidence of the members used for its sentiment for that asset.
"""
import argparse
import csv
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import fmean

from cryptosense.news import COLLECTION_START, in_study_period
from cryptosense.news.collect import DEFAULT_DATA_DIR
from cryptosense.news.label import EXPECTED_REPORTED_MODEL, PROMPT_VERSION, _read_csv

FIRST_STUDY_DAY = date(2026, 10, 25)
LAST_STUDY_DAY = date(2027, 1, 31)
MAX_FETCH_LAG = timedelta(hours=24)
DUPLICATE_WINDOW = timedelta(hours=48)
NEWSLETTER_SOURCE = "decrypt"
NEWSLETTER_MIN_BANGS = 2
MIN_COLLECTOR_HOURS = 18
MIN_MARKET_ITEMS = 5
MIN_BTC_ITEMS = 3
SCORED_ASSETS = ("market", "btc")
KEPT_PUNCTUATION = "%"  # Unicode punctuation (category P) kept by normalize_title
OUTPUT_FILE = "daily_sentiment.csv"
OUTPUT_COLUMNS = ["date", "collector_hours", "n_market", "n_btc", "s_market", "s_btc",
                  "included_h1", "included_h2", "exclusion_reason"]


def parse_utc(text):
    """Parse an ISO 8601 time with a timezone into a UTC datetime.

    Raises
    ------
    ValueError
        If the time cannot be parsed or has no timezone.
    """
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise ValueError("times must include a timezone")
    return moment.astimezone(timezone.utc)


def normalize_title(title):
    """Lowercase a title, remove Unicode punctuation except "%", collapse whitespace and trim.

    Symbols (Unicode categories S*, such as "$" or "+") are kept.
    """
    text = "".join(ch for ch in title.lower()
                   if ch in KEPT_PUNCTUATION or not unicodedata.category(ch).startswith("P"))
    return " ".join(text.split())


def is_newsletter(source, title):
    """Return True for a multi-story newsletter: a decrypt title with two or more "!"."""
    return source == NEWSLETTER_SOURCE and title.count("!") >= NEWSLETTER_MIN_BANGS


def frozen_labels(labels):
    """Return the labels from the frozen model and prompt, keyed by article id.

    Only rows with ``model_reported == EXPECTED_REPORTED_MODEL`` and
    ``prompt_version == PROMPT_VERSION`` are kept; if an article has several,
    the first row is used. Each value holds ``relevance_<asset>`` (bool),
    ``sentiment_<asset>`` (int or None) for the assets in ``SCORED_ASSETS``,
    and ``confidence`` (float).
    """
    parsed = {}
    for row in labels:
        if (row["model_reported"] != EXPECTED_REPORTED_MODEL or row["prompt_version"] != PROMPT_VERSION
                or row["article_id"] in parsed):
            continue
        label = {"confidence": float(row["confidence"])}
        for asset in SCORED_ASSETS:
            label[f"relevance_{asset}"] = row[f"relevance_{asset}"] == "true"
            raw = row[f"sentiment_{asset}"]
            label[f"sentiment_{asset}"] = int(raw) if raw != "" else None
        parsed[row["article_id"]] = label
    return parsed


def eligible_articles(articles, labels, exclude_sources=()):
    """Apply the section 4 article rules (all but duplicates).

    Parameters
    ----------
    articles : list of dict
        Rows of ``articles.csv``.
    labels : list of dict
        Rows of ``labels.csv``.
    exclude_sources : iterable of str
        Sources to drop entirely (robustness check); empty for the primary analysis.

    Returns
    -------
    list of dict
        One dict per counting article with ``id``, ``title``, ``published``
        (UTC datetime), ``day`` (UTC date) and the parsed label fields.
    """
    by_id = frozen_labels(labels)
    excluded = set(exclude_sources)
    kept = []
    for article in articles:
        if article["source"] in excluded or not in_study_period(article["published_utc"]):
            continue
        published = parse_utc(article["published_utc"])
        if parse_utc(article["fetched_utc"]) - published > MAX_FETCH_LAG:
            continue
        if is_newsletter(article["source"], article["title"]) or article["id"] not in by_id:
            continue
        kept.append({"id": article["id"], "title": article["title"], "published": published,
                     "day": published.date(), **by_id[article["id"]]})
    return kept


def _item(day, members):
    """Build one scored item from articles of the same UTC day (rule 6, clarification 3)."""
    item = {"day": day}
    for asset in SCORED_ASSETS:
        relevant = [m for m in members if m[f"relevance_{asset}"]]
        item[f"relevance_{asset}"] = bool(relevant)
        item[f"sentiment_{asset}"] = fmean(m[f"sentiment_{asset}"] for m in relevant) if relevant else None
        item[f"confidence_{asset}"] = fmean(m["confidence"] for m in relevant) if relevant else None
    return item


def build_items(articles, dedupe=True):
    """Turn eligible articles into scored items, one per duplicate group.

    With ``dedupe`` (the primary analysis), articles are grouped as described
    in clarifications 1, 3 and 4 of the module docstring; each group becomes
    one item on the day of its earliest member, built from the members
    published on that day. Without it, every article is its own item.

    Returns
    -------
    list of dict
        Items with ``day`` and, per asset in ``SCORED_ASSETS``,
        ``relevance_<asset>``, ``sentiment_<asset>`` and ``confidence_<asset>``.
    """
    ordered = sorted(articles, key=lambda a: (a["published"], a["id"]))
    if not dedupe:
        return [_item(a["day"], [a]) for a in ordered]
    groups, open_group = [], {}
    for article in ordered:
        key = normalize_title(article["title"])
        group = open_group.get(key)
        if group is not None and article["published"] - group[0]["published"] <= DUPLICATE_WINDOW:
            group.append(article)
        else:
            group = [article]
            open_group[key] = group
            groups.append(group)
    return [_item(group[0]["day"], [m for m in group if m["day"] == group[0]["day"]])
            for group in groups]


def collector_hours(runs):
    """Count, per UTC day, the hours with at least one fully successful collector run.

    A run is the set of ``runs.csv`` rows sharing a ``run_utc``; it is
    successful if every row has status ``ok``. The run counts for the hour in
    which it started.

    Returns
    -------
    dict
        ``{date: number of hours}``; days without a successful run are absent.
    """
    statuses = defaultdict(list)
    for row in runs:
        statuses[row["run_utc"]].append(row["status"])
    hours = {parse_utc(run_utc).replace(minute=0, second=0, microsecond=0)
             for run_utc, run_statuses in statuses.items()
             if all(status == "ok" for status in run_statuses)}
    return Counter(hour.date() for hour in hours)


def complete_study_days(now):
    """Return the study days that have ended by ``now``, in order.

    A day is complete once ``now`` has reached 00:00 UTC of the following day.
    """
    days, day = [], FIRST_STUDY_DAY
    while day <= LAST_STUDY_DAY and datetime.combine(day + timedelta(days=1), time(), timezone.utc) <= now:
        days.append(day)
        day += timedelta(days=1)
    return days


def _score(items, asset, weight_by_confidence):
    """Mean (optionally confidence-weighted) sentiment of the items relevant for ``asset``."""
    relevant = [i for i in items if i[f"relevance_{asset}"]]
    if not relevant:
        return None
    if not weight_by_confidence:
        return fmean(i[f"sentiment_{asset}"] for i in relevant)
    total = sum(i[f"confidence_{asset}"] for i in relevant)
    if total == 0:
        return None
    return sum(i[f"sentiment_{asset}"] * i[f"confidence_{asset}"] for i in relevant) / total


def daily_scores(items, hours, days, weight_by_confidence=False):
    """Compute the section 5 scores and section 6 coverage for each day in ``days``.

    Returns
    -------
    list of dict
        One row per day with the keys in ``OUTPUT_COLUMNS``. ``s_market`` and
        ``s_btc`` are None when the day has no relevant item. ``exclusion_reason``
        lists every failed condition (``collector_hours<18``, ``n_market<5``,
        ``n_btc<3``) separated by "; ", or is empty if the day is included for
        both H1 and H2.
    """
    by_day = defaultdict(list)
    for item in items:
        by_day[item["day"]].append(item)
    rows = []
    for day in days:
        day_items = by_day.get(day, [])
        n_market = sum(i["relevance_market"] for i in day_items)
        n_btc = sum(i["relevance_btc"] for i in day_items)
        day_hours = hours.get(day, 0)
        reasons = []
        if day_hours < MIN_COLLECTOR_HOURS:
            reasons.append(f"collector_hours<{MIN_COLLECTOR_HOURS}")
        if n_market < MIN_MARKET_ITEMS:
            reasons.append(f"n_market<{MIN_MARKET_ITEMS}")
        if n_btc < MIN_BTC_ITEMS:
            reasons.append(f"n_btc<{MIN_BTC_ITEMS}")
        rows.append({
            "date": day.isoformat(), "collector_hours": day_hours, "n_market": n_market, "n_btc": n_btc,
            "s_market": _score(day_items, "market", weight_by_confidence),
            "s_btc": _score(day_items, "btc", weight_by_confidence),
            "included_h1": day_hours >= MIN_COLLECTOR_HOURS and n_market >= MIN_MARKET_ITEMS,
            "included_h2": day_hours >= MIN_COLLECTOR_HOURS and n_btc >= MIN_BTC_ITEMS,
            "exclusion_reason": "; ".join(reasons),
        })
    return rows


def aggregate(data_dir=DEFAULT_DATA_DIR, now=None, dedupe=True, exclude_sources=(),
              weight_by_confidence=False):
    """Compute the daily rows from the files in ``data_dir / "news"`` without writing anything.

    Parameters
    ----------
    data_dir : Path
        The project data directory.
    now : callable or None
        ``now() -> datetime`` (timezone-aware); only days complete by then are
        returned. Defaults to the current UTC time.
    dedupe, exclude_sources, weight_by_confidence
        Section 9 robustness options; the defaults are the primary analysis.

    Returns
    -------
    list of dict
        See ``daily_scores``.
    """
    now = (now or (lambda: datetime.now(timezone.utc)))()
    news_dir = Path(data_dir) / "news"
    articles = eligible_articles(_read_csv(news_dir / "articles.csv"), _read_csv(news_dir / "labels.csv"),
                                 exclude_sources)
    return daily_scores(build_items(articles, dedupe), collector_hours(_read_csv(news_dir / "runs.csv")),
                        complete_study_days(now), weight_by_confidence)


def _format(value):
    """Format a value for the CSV: true/false for booleans, empty for None."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else value


def write_daily(rows, path):
    """Write the daily rows to ``path`` (overwriting it) with the columns in ``OUTPUT_COLUMNS``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows({column: _format(row[column]) for column in OUTPUT_COLUMNS} for row in rows)


def coverage_summary(rows):
    """Format coverage figures only: days so far, days included for H1 and H2, exclusion reasons.

    No sentiment value or item count is included.
    """
    lines = [f"days so far: {len(rows)}",
             f"days included for H1: {sum(row['included_h1'] for row in rows)}",
             f"days included for H2: {sum(row['included_h2'] for row in rows)}",
             "exclusion reasons:"]
    reasons = Counter(row["exclusion_reason"] for row in rows if row["exclusion_reason"])
    lines += [f"  {reason}: {count}" for reason, count in sorted(reasons.items())] or ["  none"]
    return "\n".join(lines)


def main(argv=None, now=None):
    """Command-line entry point: write ``daily_sentiment.csv`` and print coverage figures only.

    ``now`` is passed to ``aggregate`` (inject a fixed clock in tests).
    """
    parser = argparse.ArgumentParser(description="Aggregate news labels into daily sentiment.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="project data directory (default: %(default)s)")
    args = parser.parse_args(argv)
    rows = aggregate(data_dir=args.data_dir, now=now)
    write_daily(rows, Path(args.data_dir) / "news" / OUTPUT_FILE)
    print(coverage_summary(rows))


if __name__ == "__main__":
    main()
