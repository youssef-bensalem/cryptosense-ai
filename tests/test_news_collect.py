import csv
from datetime import datetime, timezone

import pandas as pd
import pytest
import requests

from cryptosense.news.collect import (
    ARTICLE_COLUMNS, RUN_COLUMNS, article_id, collect, format_summary, normalize_url,
    parse_published_utc, strip_html,
)


def make_rss(items):
    """An RSS 2.0 document; ``items`` are (link, pub_date or None, summary) tuples."""
    parts = []
    for i, (link, pub_date, summary) in enumerate(items):
        date_tag = f"<pubDate>{pub_date}</pubDate>" if pub_date is not None else ""
        parts.append(f"<item><title>Title {i}</title><link>{link}</link>{date_tag}"
                     f"<description><![CDATA[{summary}]]></description></item>")
    return ("<?xml version='1.0' encoding='utf-8'?><rss version='2.0'><channel>"
            "<title>Test</title><link>https://example.com</link><description>t</description>"
            + "".join(parts) + "</channel></rss>").encode("utf-8")


def fake_fetch(pages):
    """A fetch function serving ``pages[url]``; exceptions in ``pages`` are raised."""
    def fetch(url):
        page = pages[url]
        if isinstance(page, Exception):
            raise page
        return page
    return fetch


def fixed_now(moment="2026-10-24T12:00:00+00:00"):
    return lambda: datetime.fromisoformat(moment)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


FEED_A = {"source": "a", "url": "https://a.test/rss"}
FEED_B = {"source": "b", "url": "https://b.test/rss"}
DATE = "Sat, 24 Oct 2026 10:00:00 +0000"


# --- URL normalization and ids ---

@pytest.mark.parametrize("variant", [
    "https://example.com/news/story",
    "HTTPS://Example.COM/news/story",
    "https://example.com/news/story/",
    "https://example.com/news/story#comments",
    "https://example.com/news/story?utm_source=rss&utm_medium=feed",
    "https://example.com/news/story?UTM_Campaign=x",
    "https://example.com/news/story?ref=twitter",
    "https://example.com/news/story?source=rss",
    "  https://EXAMPLE.com/news/story/?utm_source=rss&ref=x&source=y#top  ",
])
def test_equivalent_urls_normalize_and_share_id(variant):
    assert normalize_url(variant) == "https://example.com/news/story"
    assert article_id(variant) == article_id("https://example.com/news/story")


def test_normalization_keeps_path_case_and_other_params():
    url = "https://Example.com/News/Story/?id=7&utm_source=x&page=2#frag"
    assert normalize_url(url) == "https://example.com/News/Story?id=7&page=2"
    assert article_id(url) != article_id("https://example.com/news/story?id=7&page=2")


def test_article_id_is_sha256_of_normalized_url():
    import hashlib
    expected = hashlib.sha256(b"https://example.com/a").hexdigest()
    assert article_id("https://EXAMPLE.com/a/?utm_x=1") == expected


# --- HTML stripping ---

def test_strip_html_removes_tags_and_decodes_entities():
    html = "<p>Bitcoin <b>rose</b> &amp; ETH &lt;fell&gt;.</p>\n<img src='x.png'/><br/>  Next   line"
    assert strip_html(html) == "Bitcoin rose & ETH <fell>. Next line"
    assert strip_html("<div><p>a</p>\n<p>b</p></div>") == "a b"
    assert strip_html(None) == ""


def test_summaries_are_stored_without_html(tmp_path):
    rss = make_rss([("https://a.test/1", DATE, "<p>Price <a href='https://x'>up</a> &amp; out</p>")])
    collect([FEED_A], tmp_path, fake_fetch({FEED_A["url"]: rss}), fixed_now())
    [row] = read_csv(tmp_path / "news" / "articles.csv")
    assert row["summary"] == "Price up & out"


# --- timestamps ---

@pytest.mark.parametrize("raw, expected", [
    ("Sat, 24 Oct 2026 18:00:00 +0200", "2026-10-24T16:00:00Z"),
    ("Sat, 24 Oct 2026 21:30:00 -0500", "2026-10-25T02:30:00Z"),
    ("Sat, 24 Oct 2026 16:00:00 +0000", "2026-10-24T16:00:00Z"),
    ("Sat, 24 Oct 2026 16:00:00 GMT", "2026-10-24T16:00:00Z"),
    ("2026-10-24T18:00:00+02:00", "2026-10-24T16:00:00Z"),
    ("2026-10-24T16:00:00Z", "2026-10-24T16:00:00Z"),
])
def test_publication_times_convert_to_utc(raw, expected):
    assert parse_published_utc(raw).strftime("%Y-%m-%dT%H:%M:%SZ") == expected
    assert parse_published_utc(raw).utcoffset().total_seconds() == 0


def test_feed_with_offsets_is_stored_in_utc(tmp_path):
    rss = make_rss([
        ("https://a.test/1", "Sat, 24 Oct 2026 18:00:00 +0200", "s"),
        ("https://a.test/2", "Sat, 24 Oct 2026 21:30:00 -0500", "s"),
        ("https://a.test/3", "Sat, 24 Oct 2026 16:00:00 +0000", "s"),
    ])
    collect([FEED_A], tmp_path, fake_fetch({FEED_A["url"]: rss}), fixed_now())
    rows = read_csv(tmp_path / "news" / "articles.csv")
    assert [r["published_utc"] for r in rows] == [
        "2026-10-24T16:00:00Z", "2026-10-25T02:30:00Z", "2026-10-24T16:00:00Z"]
    assert all(r["fetched_utc"] == "2026-10-24T12:00:00Z" for r in rows)


@pytest.mark.parametrize("raw", [
    None, "", "   ", "yesterday", "Sat, 99 Foo 2026", "2026-13-45T00:00:00Z",
    "Sat, 24 Oct 2026 16:00:00",      # no timezone: not assumed to be UTC
    "Sat, 24 Oct 2026 16:00:00 -0000",  # RFC 822 "timezone unknown"
    "2026-10-24T16:00:00",            # ISO without timezone
])
def test_missing_or_unparseable_times_return_none(raw):
    assert parse_published_utc(raw) is None


def test_articles_without_valid_time_are_skipped_and_counted(tmp_path):
    rss = make_rss([
        ("https://a.test/ok", DATE, "s"),
        ("https://a.test/missing", None, "s"),
        ("https://a.test/garbage", "not a date", "s"),
        ("https://a.test/naive", "Sat, 24 Oct 2026 16:00:00", "s"),
    ])
    [run] = collect([FEED_A], tmp_path, fake_fetch({FEED_A["url"]: rss}), fixed_now())
    assert (run["status"], run["new"], run["duplicate"], run["skipped"]) == ("ok", 1, 0, 3)
    assert [r["url"] for r in read_csv(tmp_path / "news" / "articles.csv")] == ["https://a.test/ok"]
    [logged] = read_csv(tmp_path / "news" / "runs.csv")
    assert logged["skipped"] == "3"


# --- dedup ---

def test_dedup_within_one_run_keeps_first_seen(tmp_path):
    rss_a = make_rss([
        ("https://a.test/story", DATE, "first"),
        ("https://A.test/story/?utm_source=rss", DATE, "same story again"),
    ])
    rss_b = make_rss([("https://a.test/story#x", DATE, "from feed b"),
                      ("https://b.test/other", DATE, "other")])
    runs = collect([FEED_A, FEED_B], tmp_path,
                   fake_fetch({FEED_A["url"]: rss_a, FEED_B["url"]: rss_b}), fixed_now())
    assert [(r["new"], r["duplicate"]) for r in runs] == [(1, 1), (1, 1)]
    rows = read_csv(tmp_path / "news" / "articles.csv")
    assert [(r["source"], r["summary"]) for r in rows] == [("a", "first"), ("b", "other")]
    assert len({r["id"] for r in rows}) == 2


def test_dedup_across_two_runs(tmp_path):
    first = make_rss([("https://a.test/1", DATE, "one"), ("https://a.test/2", DATE, "two")])
    second = make_rss([("https://a.test/2?ref=home", DATE, "two again"),
                       ("https://a.test/3", DATE, "three")])
    collect([FEED_A], tmp_path, fake_fetch({FEED_A["url"]: first}), fixed_now())
    [run] = collect([FEED_A], tmp_path, fake_fetch({FEED_A["url"]: second}),
                    fixed_now("2026-10-24T13:00:00+00:00"))
    assert (run["new"], run["duplicate"]) == (1, 1)
    rows = read_csv(tmp_path / "news" / "articles.csv")
    assert [r["summary"] for r in rows] == ["one", "two", "three"]
    assert [r["fetched_utc"] for r in rows] == [
        "2026-10-24T12:00:00Z", "2026-10-24T12:00:00Z", "2026-10-24T13:00:00Z"]
    assert len(read_csv(tmp_path / "news" / "runs.csv")) == 2


def test_later_runs_never_rewrite_earlier_rows(tmp_path):
    """Truncation check: what was stored at time t is unchanged by any later run."""
    path = tmp_path / "news" / "articles.csv"
    collect([FEED_A], tmp_path, fake_fetch({FEED_A["url"]: make_rss([("https://a.test/1", DATE, "x")])}),
            fixed_now())
    before = path.read_bytes()
    later = make_rss([("https://a.test/1", "Sun, 25 Oct 2026 10:00:00 +0000", "edited"),
                      ("https://a.test/2", DATE, "y")])
    collect([FEED_A], tmp_path, fake_fetch({FEED_A["url"]: later}), fixed_now("2026-10-25T12:00:00+00:00"))
    assert path.read_bytes().startswith(before)
    assert len(read_csv(path)) == 2


# --- failing feeds ---

@pytest.mark.parametrize("failure", [
    requests.ConnectionError("connection refused"),
    requests.Timeout("timed out"),
    requests.HTTPError("503 Server Error"),
    b"<?xml version='1.0'?><rss version='2.0'><channel><item><title>broken</ite",
    b"<html><body>Not a feed</body></html>",
])
def test_failing_feed_is_logged_and_others_still_save(tmp_path, failure):
    feed_c = {"source": "c", "url": "https://c.test/rss"}
    pages = {
        FEED_A["url"]: make_rss([("https://a.test/1", DATE, "s")]),
        FEED_B["url"]: failure,
        feed_c["url"]: make_rss([("https://c.test/1", DATE, "s")]),
    }
    runs = collect([FEED_A, FEED_B, feed_c], tmp_path, fake_fetch(pages), fixed_now())
    assert [r["status"] for r in runs] == ["ok", "failed", "ok"]

    logged = read_csv(tmp_path / "news" / "runs.csv")
    assert list(logged[0]) == RUN_COLUMNS
    assert [(r["source"], r["status"]) for r in logged] == [("a", "ok"), ("b", "failed"), ("c", "ok")]
    assert logged[1]["error"] != "" and logged[1]["new"] == "0"
    assert all(r["error"] == "" for r in (logged[0], logged[2]))
    assert {r["run_utc"] for r in logged} == {"2026-10-24T12:00:00Z"}
    assert [r["source"] for r in read_csv(tmp_path / "news" / "articles.csv")] == ["a", "c"]


def test_http_error_status_from_fetch_feed_raises(monkeypatch):
    from cryptosense.news import collect as module

    class FakeResponse:
        content = b""

        def raise_for_status(self):
            raise requests.HTTPError("404 Client Error")

    seen = {}

    def fake_get(url, timeout, headers):
        seen.update(url=url, timeout=timeout, headers=headers)
        return FakeResponse()

    monkeypatch.setattr(module.requests, "get", fake_get)
    with pytest.raises(requests.HTTPError):
        module.fetch_feed("https://x.test/rss")
    assert seen["timeout"] == 20
    assert "CryptoSense" in seen["headers"]["User-Agent"]


# --- CSV round trip and CLI summary ---

def test_csv_round_trip_keeps_identical_timestamps(tmp_path):
    rss = make_rss([
        ("https://a.test/1", "Sat, 24 Oct 2026 18:00:00 +0200", "s"),
        ("https://a.test/2", "Sat, 24 Oct 2026 21:30:59 -0500", "s"),
    ])
    collect([FEED_A], tmp_path, fake_fetch({FEED_A["url"]: rss}), fixed_now("2026-10-24T12:34:56+00:00"))
    path = tmp_path / "news" / "articles.csv"

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(df.columns) == ARTICLE_COLUMNS
    assert list(df["published_utc"]) == ["2026-10-24T16:00:00Z", "2026-10-25T02:30:59Z"]

    parsed = pd.to_datetime(df["published_utc"], utc=True, format="ISO8601")
    assert list(parsed) == [pd.Timestamp("2026-10-24T16:00:00Z"), pd.Timestamp("2026-10-25T02:30:59Z")]
    assert list(parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")) == list(df["published_utc"])

    copy = tmp_path / "copy.csv"
    df.to_csv(copy, index=False)
    assert pd.read_csv(copy, dtype=str, keep_default_na=False).equals(df)


def test_format_summary_totals():
    runs = [
        {"source": "a", "status": "ok", "new": 3, "duplicate": 1, "skipped": 2},
        {"source": "b", "status": "failed", "new": 0, "duplicate": 0, "skipped": 0},
    ]
    lines = format_summary(runs).splitlines()
    assert lines[-1].split() == ["total", "3", "1", "2", "1"]
    assert lines[2].split() == ["b", "0", "0", "0", "1"]
