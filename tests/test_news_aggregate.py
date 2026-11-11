import ast
import csv
import hashlib
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

import pytest

import cryptosense.news
from cryptosense.news import aggregate as module
from cryptosense.news.aggregate import (
    OUTPUT_COLUMNS, aggregate, build_items, collector_hours, complete_study_days, daily_scores,
    eligible_articles, frozen_labels, is_newsletter, normalize_title,
)
from cryptosense.news.collect import ARTICLE_COLUMNS, RUN_COLUMNS
from cryptosense.news.feeds import FEEDS
from cryptosense.news.label import LABEL_COLUMNS

SOURCES = [feed["source"] for feed in FEEDS]
OTHER_PROMPT = "3245177aefd0"


def utc(text):
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def fmt(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def art(article_id, published, source="coindesk", title=None, fetched=None):
    """An articles.csv row; fetched one hour after publication unless given."""
    fetched = fetched or fmt(utc(published) + timedelta(hours=1))
    return {"id": article_id, "source": source, "title": title or f"Story {article_id}",
            "summary": "", "url": f"https://x.test/{article_id}", "published_utc": published,
            "fetched_utc": fetched}


def lab(article_id, market=1, btc=None, confidence=0.8, model="openai/gpt-oss-120b",
        prompt="affefb1b22ff"):
    """A labels.csv row; ``market``/``btc`` None means not relevant for that asset."""
    def relevance(value):
        return "false" if value is None else "true"

    def sentiment(value):
        return "" if value is None else str(value)

    return {"article_id": article_id, "model_requested": "gpt-oss-120b", "model_reported": model,
            "prompt_version": prompt, "labeled_utc": "2026-10-28T09:00:00Z",
            "relevance_btc": relevance(btc), "relevance_sol": "false", "relevance_market": relevance(market),
            "sentiment_btc": sentiment(btc), "sentiment_sol": "", "sentiment_market": sentiment(market),
            "event_type": "other", "confidence": str(confidence)}


def runs_for(day, hours=range(24), failed=()):
    """Collector runs at minute 5 of each given hour of ``day`` (ISO date); hours in ``failed``
    have one failed source."""
    rows = []
    for hour in hours:
        run_utc = f"{day}T{hour:02d}:05:00Z"
        for position, source in enumerate(SOURCES):
            status = "failed" if hour in failed and position == 0 else "ok"
            rows.append({"run_utc": run_utc, "source": source, "status": status, "new": 0,
                         "duplicate": 0, "skipped": 0, "error": ""})
    return rows


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_inputs(tmp_path, articles, labels, runs):
    news = tmp_path / "news"
    write_csv(news / "articles.csv", ARTICLE_COLUMNS, articles)
    write_csv(news / "labels.csv", LABEL_COLUMNS, labels)
    write_csv(news / "runs.csv", RUN_COLUMNS, runs)


def clock(text):
    return lambda: utc(text)


def by_date(rows):
    return {row["date"]: row for row in rows}


def ids(articles):
    return {a["id"] for a in articles}


# --- article rules -------------------------------------------------------------------------

def test_cutoff_one_second_before_is_excluded_and_at_or_after_is_kept():
    articles = [art("before", "2026-10-24T22:01:52Z"), art("at", "2026-10-24T22:01:53Z"),
                art("after", "2026-10-24T22:01:54Z")]
    labels = [lab(a["id"]) for a in articles]
    assert ids(eligible_articles(articles, labels)) == {"at", "after"}


def test_collection_start_is_imported_not_retyped():
    assert module.COLLECTION_START is cryptosense.news.COLLECTION_START
    source = open(module.__file__, encoding="utf-8").read()
    assert cryptosense.news.COLLECTION_START not in source


def test_day_is_utc_date_of_publication():
    articles = [art("late", "2026-10-25T23:59:59Z"), art("midnight", "2026-10-26T00:00:00Z"),
                art("offset", "2026-10-26T01:30:00+02:00")]
    days = {a["id"]: a["day"] for a in eligible_articles(articles, [lab(a["id"]) for a in articles])}
    assert days == {"late": date(2026, 10, 25), "midnight": date(2026, 10, 26),
                    "offset": date(2026, 10, 25)}


def test_fetch_lag_exactly_24_hours_is_kept_one_second_more_is_excluded():
    articles = [art("ok", "2026-10-25T10:00:00Z", fetched="2026-10-26T10:00:00Z"),
                art("late", "2026-10-25T10:00:00Z", fetched="2026-10-26T10:00:01Z")]
    assert ids(eligible_articles(articles, [lab("ok"), lab("late")])) == {"ok"}


def test_newsletter_rule_only_for_decrypt_and_two_or_more_bangs():
    assert is_newsletter("decrypt", "Crypto is Green! BTC up!")
    assert is_newsletter("decrypt", "A! B! C!")
    assert not is_newsletter("decrypt", "Bitcoin hits a record!")
    assert not is_newsletter("decrypt", "No exclamation")
    assert not is_newsletter("coindesk", "Crypto is Green! BTC up!")
    articles = [art("news", "2026-10-25T10:00:00Z", "decrypt", "Crypto is Green! BTC up!"),
                art("one", "2026-10-25T10:00:00Z", "decrypt", "Bitcoin hits a record!"),
                art("cd", "2026-10-25T10:00:00Z", "coindesk", "Crypto is Green! BTC up!")]
    assert ids(eligible_articles(articles, [lab(a["id"]) for a in articles])) == {"one", "cd"}


def test_labels_from_another_model_or_prompt_are_ignored():
    articles = [art("m", "2026-10-25T10:00:00Z"), art("p", "2026-10-25T11:00:00Z"),
                art("both", "2026-10-25T12:00:00Z")]
    labels = [lab("m", model="google/gemini-2.5-flash"), lab("p", prompt=OTHER_PROMPT),
              lab("both", market=-2, prompt=OTHER_PROMPT), lab("both", market=2)]
    kept = eligible_articles(articles, labels)
    assert ids(kept) == {"both"}
    assert kept[0]["sentiment_market"] == 2


def test_frozen_label_values_match_the_preregistration():
    assert module.EXPECTED_REPORTED_MODEL == "openai/gpt-oss-120b"
    assert module.PROMPT_VERSION == "affefb1b22ff"


def test_first_frozen_label_is_used_when_an_article_has_several():
    labels = frozen_labels([lab("a", market=1), lab("a", market=-1)])
    assert labels["a"]["sentiment_market"] == 1


def test_unlabeled_article_does_not_count():
    assert eligible_articles([art("a", "2026-10-25T10:00:00Z")], []) == []


def test_excluded_sources_are_dropped():
    articles = [art("cd", "2026-10-25T10:00:00Z", "coindesk"), art("ct", "2026-10-25T10:00:00Z", "cointelegraph")]
    kept = eligible_articles(articles, [lab("cd"), lab("ct")], exclude_sources=("coindesk",))
    assert ids(kept) == {"ct"}


# --- duplicates ------------------------------------------------------------------------------

def test_normalize_title_ignores_case_punctuation_and_whitespace():
    assert normalize_title("Bitcoin Hits $100K!") == normalize_title("bitcoin hits $100k")
    assert normalize_title("SEC’s “new” rule — explained") == normalize_title("secs new rule explained")
    assert normalize_title("Price  up 5%") == "price up 5%"
    assert normalize_title("Price up 5%") != normalize_title("Price up 5")
    assert normalize_title("  A #1 & B* @ C/D_E ") == "a 1 b cde"
    assert normalize_title("BTC +5 = $100K | ETH ~ ^") == "btc +5 = $100k | eth ~ ^"
    assert normalize_title("ETF approved") != normalize_title("ETF rejected")


def items_for(articles, labels, dedupe=True):
    return build_items(eligible_articles(articles, labels), dedupe)


def test_case_and_punctuation_differences_are_grouped_same_day_mean():
    articles = [art("a", "2026-10-25T10:00:00Z", "coindesk", "Bitcoin Hits $100K!"),
                art("b", "2026-10-25T12:00:00Z", "decrypt", "bitcoin hits $100k")]
    items = items_for(articles, [lab("a", market=2), lab("b", market=1)])
    assert len(items) == 1
    assert items[0]["sentiment_market"] == 1.5


def test_within_48_hours_is_grouped_49_hours_is_not():
    within = [art("a", "2026-10-25T10:00:00Z", title="Same story"),
              art("b", "2026-10-27T10:00:00Z", title="Same story")]
    beyond = [art("a", "2026-10-25T10:00:00Z", title="Same story"),
              art("b", "2026-10-27T11:00:00Z", title="Same story")]
    labels = [lab("a"), lab("b")]
    assert [i["day"] for i in items_for(within, labels)] == [date(2026, 10, 25)]
    assert [i["day"] for i in items_for(beyond, labels)] == [date(2026, 10, 25), date(2026, 10, 27)]


def test_duplicate_on_next_day_is_dropped_and_does_not_change_earlier_day():
    articles = [art("a", "2026-10-25T20:00:00Z", title="Same story"),
                art("b", "2026-10-26T02:00:00Z", title="Same story")]
    items = items_for(articles, [lab("a", market=1), lab("b", market=-2)])
    assert len(items) == 1
    assert items[0]["day"] == date(2026, 10, 25)
    assert items[0]["sentiment_market"] == 1


def test_chain_groups_within_48_hours_of_earliest_member():
    articles = [art("a", "2026-10-25T00:00:00Z", title="Same story"),
                art("b", "2026-10-26T16:00:00Z", title="Same story"),
                art("c", "2026-10-28T08:00:00Z", title="Same story")]
    items = items_for(articles, [lab("a"), lab("b"), lab("c")])
    assert [i["day"] for i in items] == [date(2026, 10, 25), date(2026, 10, 28)]


def test_group_relevance_any_member_and_mean_over_relevant_members_per_asset():
    articles = [art("a", "2026-10-25T10:00:00Z", title="Same story"),
                art("b", "2026-10-25T11:00:00Z", title="Same story"),
                art("c", "2026-10-25T12:00:00Z", title="Same story")]
    labels = [lab("a", market=2, btc=None, confidence=0.9), lab("b", market=None, btc=-1, confidence=0.5),
              lab("c", market=0, btc=None, confidence=0.7)]
    [item] = items_for(articles, labels)
    assert item["relevance_market"] and item["relevance_btc"]
    assert item["sentiment_market"] == 1
    assert item["sentiment_btc"] == -1
    assert item["confidence_market"] == pytest.approx(0.8)
    assert item["confidence_btc"] == 0.5


def test_without_dedupe_every_article_is_an_item():
    articles = [art("a", "2026-10-25T10:00:00Z", title="Same story"),
                art("b", "2026-10-25T11:00:00Z", title="Same story")]
    assert len(items_for(articles, [lab("a"), lab("b")], dedupe=False)) == 2


def test_rule_failing_article_is_not_a_group_member():
    articles = [art("a", "2026-10-25T10:00:00Z", title="Same story", fetched="2026-10-27T10:00:00Z"),
                art("b", "2026-10-26T09:00:00Z", title="Same story")]
    items = items_for(articles, [lab("a"), lab("b")])
    assert [i["day"] for i in items] == [date(2026, 10, 26)]


# --- daily scores ----------------------------------------------------------------------------

DAY = [date(2026, 10, 25)]
FULL = {date(2026, 10, 25): 24}


def test_relevance_filtering_means_and_counts():
    articles = [art(str(n), f"2026-10-25T{10 + n:02d}:00:00Z") for n in range(5)]
    labels = [lab("0", market=2, btc=1), lab("1", market=1, btc=-1), lab("2", market=0),
              lab("3", market=None, btc=None), lab("4", market=None, btc=1)]
    [row] = daily_scores(items_for(articles, labels), FULL, DAY)
    assert (row["n_market"], row["s_market"]) == (3, 1.0)
    assert row["n_btc"] == 3
    assert row["s_btc"] == pytest.approx(1 / 3)


def test_day_without_relevant_items_has_no_score():
    [row] = daily_scores([], FULL, DAY)
    assert (row["n_market"], row["n_btc"], row["s_market"], row["s_btc"]) == (0, 0, None, None)


def test_weight_by_confidence():
    articles = [art("a", "2026-10-25T10:00:00Z"), art("b", "2026-10-25T11:00:00Z")]
    items = items_for(articles, [lab("a", market=2, confidence=0.75), lab("b", market=-2, confidence=0.25)])
    assert daily_scores(items, FULL, DAY)[0]["s_market"] == 0
    assert daily_scores(items, FULL, DAY, weight_by_confidence=True)[0]["s_market"] == 1


# --- coverage --------------------------------------------------------------------------------

def test_collector_hours_18_included_17_excluded():
    hours = collector_hours(runs_for("2026-10-25", range(18)) + runs_for("2026-10-26", range(17)))
    assert hours[date(2026, 10, 25)] == 18
    assert hours[date(2026, 10, 26)] == 17
    days = [date(2026, 10, 25), date(2026, 10, 26)]
    articles = [art(f"{d}-{n}", f"{d}T0{n}:00:00Z") for d in ("2026-10-25", "2026-10-26") for n in range(5)]
    rows = daily_scores(items_for(articles, [lab(a["id"], btc=1) for a in articles]), hours, days)
    assert [(r["included_h1"], r["included_h2"]) for r in rows] == [(True, True), (False, False)]
    assert rows[1]["exclusion_reason"] == "collector_hours<18"


def test_run_with_one_failed_source_does_not_count():
    hours = collector_hours(runs_for("2026-10-25", range(3), failed={1}))
    assert hours[date(2026, 10, 25)] == 2


def test_hour_counts_if_any_run_in_it_is_fully_ok_and_hours_are_not_double_counted():
    runs = runs_for("2026-10-25", [5], failed={5})
    runs += [dict(r, run_utc="2026-10-25T05:35:00Z", status="ok") for r in runs_for("2026-10-25", [5])]
    runs += [dict(r, run_utc="2026-10-25T05:50:00Z") for r in runs_for("2026-10-25", [5])]
    assert collector_hours(runs) == {date(2026, 10, 25): 1}


def test_item_thresholds_5_market_and_3_btc():
    def row_for(n_market, n_btc):
        articles = [art(str(n), f"2026-10-25T{n:02d}:00:00Z") for n in range(max(n_market, n_btc))]
        labels = [lab(str(n), market=1 if n < n_market else None, btc=1 if n < n_btc else None)
                  for n in range(max(n_market, n_btc))]
        return daily_scores(items_for(articles, labels), FULL, DAY)[0]

    assert row_for(5, 3)["included_h1"] and row_for(5, 3)["included_h2"]
    assert row_for(5, 3)["exclusion_reason"] == ""
    four = row_for(4, 3)
    assert (four["included_h1"], four["included_h2"], four["exclusion_reason"]) == (False, True, "n_market<5")
    two = row_for(5, 2)
    assert (two["included_h1"], two["included_h2"], two["exclusion_reason"]) == (True, False, "n_btc<3")


# --- study days ------------------------------------------------------------------------------

def test_only_complete_study_days_from_2026_10_25():
    assert complete_study_days(utc("2026-10-25T23:59:59Z")) == []
    assert complete_study_days(utc("2026-10-26T00:00:00Z")) == [date(2026, 10, 25)]
    assert complete_study_days(utc("2026-10-28T12:00:00Z")) == [date(2026, 10, 25), date(2026, 10, 26),
                                                                date(2026, 10, 27)]
    days = complete_study_days(utc("2027-06-01T00:00:00Z"))
    assert (days[0], days[-1], len(days)) == (date(2026, 10, 25), date(2027, 1, 31), 99)


def test_aggregate_never_returns_an_incomplete_day_or_a_backlog_day(tmp_path):
    articles = [art("eve", "2026-10-24T23:00:00Z"), art("d1", "2026-10-25T10:00:00Z"),
                art("today", "2026-10-26T10:00:00Z")]
    write_inputs(tmp_path, articles, [lab(a["id"]) for a in articles], runs_for("2026-10-25"))
    rows = aggregate(tmp_path, now=clock("2026-10-26T12:00:00Z"))
    assert [r["date"] for r in rows] == ["2026-10-25"]
    assert rows[0]["n_market"] == 1


# --- no look-ahead ---------------------------------------------------------------------------

def base_inputs():
    articles = [art("a", "2026-10-25T08:00:00Z", title="Same story"), art("b", "2026-10-25T09:00:00Z"),
                art("c", "2026-10-25T23:59:59Z")]
    labels = [lab("a", market=2, btc=1), lab("b", market=-1, btc=None), lab("c", market=1, btc=2)]
    return articles, labels, runs_for("2026-10-25", range(20))


def test_later_articles_labels_and_runs_never_change_earlier_day(tmp_path):
    articles, labels, runs = base_inputs()
    write_inputs(tmp_path / "base", articles, labels, runs)
    base = by_date(aggregate(tmp_path / "base", now=clock("2026-10-26T00:00:00Z")))

    later_articles = [art("dup", "2026-10-26T00:00:00Z", title="Same story"),
                      art("x", "2026-10-26T10:00:00Z"), art("y", "2026-10-27T10:00:00Z")]
    later_labels = [lab("dup", market=-2, btc=-2), lab("x", market=-2), lab("y", market=-2)]
    later_runs = runs_for("2026-10-26") + runs_for("2026-10-27", failed={3})
    write_inputs(tmp_path / "more", articles + later_articles, labels + later_labels, runs + later_runs)
    more = by_date(aggregate(tmp_path / "more", now=clock("2026-10-28T00:00:00Z")))

    assert more["2026-10-25"] == base["2026-10-25"]
    assert base["2026-10-25"]["n_market"] == 3
    assert set(more) == {"2026-10-25", "2026-10-26", "2026-10-27"}


def test_truncating_inputs_after_day_gives_identical_rows(tmp_path):
    articles, labels, runs = base_inputs()
    articles += [art(f"n{n}", f"2026-10-{26 + n}T12:00:00Z") for n in range(3)]
    labels += [lab(f"n{n}", market=n - 1, btc=1) for n in range(3)]
    runs += runs_for("2026-10-26") + runs_for("2026-10-27") + runs_for("2026-10-28")
    write_inputs(tmp_path / "full", articles, labels, runs)
    full = by_date(aggregate(tmp_path / "full", now=clock("2026-10-29T00:00:00Z")))

    cutoff = utc("2026-10-27T00:00:00Z")
    kept = [a for a in articles if utc(a["published_utc"]) < cutoff]
    write_inputs(tmp_path / "cut", kept, [l for l in labels if l["article_id"] in ids(kept)],
                 [r for r in runs if utc(r["run_utc"]) < cutoff])
    cut = by_date(aggregate(tmp_path / "cut", now=clock("2026-10-27T00:00:00Z")))

    assert list(cut) == ["2026-10-25", "2026-10-26"]
    for day in cut:
        assert cut[day] == full[day]


# --- isolation from prices -------------------------------------------------------------------

def test_module_imports_no_price_loading_function():
    tree = ast.parse(open(module.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
            imported |= {alias.name for alias in node.names}
    forbidden = {"cryptosense.data", "load_processed", "fetch_ohlcv_history", "ccxt", "pandas"}
    assert not imported & forbidden
    assert not any(name.startswith("cryptosense.") and not name.startswith("cryptosense.news")
                   for name in imported)


def test_importing_module_does_not_load_price_code():
    code = ("import sys, cryptosense.news.aggregate; "
            "print(sorted(m for m in ('cryptosense.data', 'ccxt') if m in sys.modules))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


# --- output and CLI --------------------------------------------------------------------------

def cli_inputs(tmp_path):
    articles = [art(str(n), f"2026-10-25T{n:02d}:30:00Z") for n in range(6)]
    articles += [art("d2", "2026-10-26T10:00:00Z")]
    labels = [lab(str(n), market=1 if n < 4 else -1, btc=2 if n < 3 else None) for n in range(6)]
    labels += [lab("d2", market=1)]
    runs = runs_for("2026-10-25") + runs_for("2026-10-26", range(10))
    write_inputs(tmp_path, articles, labels, runs)


def test_cli_writes_file_and_prints_coverage_only(tmp_path, capsys):
    cli_inputs(tmp_path)
    module.main(["--data-dir", str(tmp_path)], now=clock("2026-10-27T00:00:00Z"))
    out = capsys.readouterr().out

    with (tmp_path / "news" / "daily_sentiment.csv").open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert list(written[0]) == OUTPUT_COLUMNS
    first = written[0]
    assert (first["date"], first["collector_hours"], first["n_market"], first["n_btc"]) == ("2026-10-25", "24", "6", "3")
    assert float(first["s_market"]) == pytest.approx(1 / 3)
    assert float(first["s_btc"]) == 2
    assert (first["included_h1"], first["included_h2"], first["exclusion_reason"]) == ("true", "true", "")
    assert written[1]["exclusion_reason"] == "collector_hours<18; n_market<5; n_btc<3"
    assert written[1]["s_btc"] == ""

    assert out.splitlines() == ["days so far: 2", "days included for H1: 1", "days included for H2: 1",
                                "exclusion reasons:", "  collector_hours<18; n_market<5; n_btc<3: 1"]
    assert "0.33" not in out and "s_market" not in out and "s_btc" not in out
    assert not re.search(r"-?\d+\.\d", out)


def test_input_files_are_not_modified(tmp_path):
    cli_inputs(tmp_path)
    paths = [tmp_path / "news" / name for name in ("articles.csv", "labels.csv", "runs.csv")]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    module.main(["--data-dir", str(tmp_path)], now=clock("2026-10-28T00:00:00Z"))
    aggregate(tmp_path, now=clock("2026-10-28T00:00:00Z"), dedupe=False, exclude_sources=("coindesk",),
              weight_by_confidence=True)
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == before


def test_defaults_match_primary_analysis(tmp_path):
    cli_inputs(tmp_path)
    now = clock("2026-10-28T00:00:00Z")
    assert aggregate(tmp_path, now=now) == aggregate(tmp_path, now=now, dedupe=True, exclude_sources=(),
                                                    weight_by_confidence=False)
