import csv
import json
import logging
from datetime import datetime

import pytest
import requests

from cryptosense.news import label as module
from cryptosense.news.collect import ARTICLE_COLUMNS
from cryptosense.news.label import (
    EVENT_TYPES, EXPECTED_REPORTED_MODEL, LABEL_COLUMNS, MODEL, PROMPT_TEMPLATE, PROMPT_VERSION,
    RUN_COLUMNS,
    build_messages, label_articles, parse_reply, prompt_version, read_env_key, strip_boilerplate,
)

KEY = "sk-test-SECRET-key-0123456789"

VALID = {"relevance_btc": True, "relevance_sol": False, "relevance_market": True,
         "sentiment_btc": 1, "sentiment_sol": None, "sentiment_market": -1,
         "event_type": "institutional", "confidence": 0.8}


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body

    def json(self):
        if isinstance(self._body, (bytes, str)):
            return json.loads(self._body)
        return self._body


def reply(content, model=EXPECTED_REPORTED_MODEL):
    """A 200 chat-completions response whose message is ``content``."""
    if not isinstance(content, str):
        content = json.dumps(content)
    return FakeResponse(200, {"model": model, "choices": [{"message": {"role": "assistant",
                                                                       "content": content}}]})


class FakeSession:
    """Returns queued responses in order; exceptions in the queue are raised."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def fixed_now(moment="2026-10-25T09:00:00+00:00"):
    return lambda: datetime.fromisoformat(moment)


def write_articles(tmp_path, articles):
    """Write an articles.csv; ``articles`` are (id, published_utc) pairs."""
    path = tmp_path / "news" / "articles.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ARTICLE_COLUMNS)
        writer.writeheader()
        for article_id, published in articles:
            writer.writerow({"id": article_id, "source": "src", "title": f"Title {article_id}",
                             "summary": f"Summary {article_id}", "url": f"https://x.test/{article_id}",
                             "published_utc": published, "fetched_utc": "2026-10-24T22:00:00Z"})


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(tmp_path, responses, **kwargs):
    session = FakeSession(responses)
    result = label_articles(KEY, tmp_path, session=session, now=fixed_now(),
                            sleep=lambda s: None, **kwargs)
    return result, session


ONE = [("a1", "2026-10-24T10:00:00Z")]


# --- valid replies ---

def test_valid_reply_is_stored_with_model_and_prompt_version(tmp_path):
    write_articles(tmp_path, ONE)
    result, session = run(tmp_path, [reply(VALID)])

    assert (result["attempted"], result["labeled"], result["invalid"], result["failed"]) == (1, 1, 0, 0)
    [row] = read_csv(tmp_path / "news" / "labels.csv")
    assert list(row) == LABEL_COLUMNS
    assert row == {
        "article_id": "a1", "model_requested": MODEL, "model_reported": EXPECTED_REPORTED_MODEL,
        "prompt_version": PROMPT_VERSION, "labeled_utc": "2026-10-25T09:00:00Z",
        "relevance_btc": "true", "relevance_sol": "false", "relevance_market": "true",
        "sentiment_btc": "1", "sentiment_sol": "", "sentiment_market": "-1",
        "event_type": "institutional", "confidence": "0.8",
    }

    [call] = session.calls
    assert call["url"] == "http://localhost:31415/v1/chat/completions"
    assert call["timeout"] == 60
    assert call["json"]["model"] == "gpt-oss-120b"
    assert call["json"]["temperature"] == 0


def test_model_constant_is_pinned_not_a_router():
    assert MODEL == "gpt-oss-120b"
    assert MODEL not in {"auto", "default", "fusion", "free-router", "auto-route", "gemini-flash"}


@pytest.mark.parametrize("reported", ["gemini-3.5-flash-lite", "gpt-oss-120b", ""])
def test_mismatched_reported_model_is_rejected_not_stored_and_stops_run(tmp_path, caplog, reported):
    write_articles(tmp_path, [("a1", "2026-10-24T10:00:00Z"), ("a2", "2026-10-24T11:00:00Z")])
    with caplog.at_level(logging.WARNING):
        result, session = run(tmp_path, [reply(VALID, model=reported), reply(VALID)])

    assert len(session.calls) == 1
    assert (result["attempted"], result["labeled"], result["invalid"], result["failed"]) == (1, 0, 0, 1)
    assert "mismatch" in result["error"] and EXPECTED_REPORTED_MODEL in result["error"]
    assert read_csv(tmp_path / "news" / "labels.csv") == []
    [logged] = read_csv(tmp_path / "news" / "label_runs.csv")
    assert logged["failed"] == "1" and "mismatch" in logged["error"]
    assert "mismatch" in caplog.text and repr(reported) in caplog.text


def test_reasoning_field_is_ignored_and_only_content_is_parsed(tmp_path):
    write_articles(tmp_path, ONE)
    message = {"role": "assistant", "content": json.dumps(VALID),
               "reasoning": "Thinking... {\"relevance_btc\": false} not the answer"}
    response = FakeResponse(200, {"model": EXPECTED_REPORTED_MODEL, "choices": [{"message": message}]})
    result, _ = run(tmp_path, [response])
    assert result["labeled"] == 1
    [row] = read_csv(tmp_path / "news" / "labels.csv")
    assert row["relevance_btc"] == "true" and row["event_type"] == "institutional"


@pytest.mark.parametrize("label", [
    VALID,
    {**VALID, "relevance_btc": False, "sentiment_btc": None},
    {**VALID, "relevance_sol": True, "sentiment_sol": -2},
    {**VALID, "sentiment_btc": 2, "confidence": 1},
    {**VALID, "confidence": 0},
    *({**VALID, "event_type": event} for event in EVENT_TYPES),
])
def test_valid_labels_are_accepted(label):
    assert parse_reply(json.dumps(label)) == (label, None)


# --- invalid replies ---

def without(field):
    return {k: v for k, v in VALID.items() if k != field}


INVALID = {
    "not json": "Sure! Here is the label: positive",
    "json in code fence": "```json\n" + json.dumps(VALID) + "\n```",
    "truncated json": json.dumps(VALID)[:-5],
    "json array": json.dumps([VALID]),
    "missing field": json.dumps(without("event_type")),
    "missing confidence": json.dumps(without("confidence")),
    "extra field": json.dumps({**VALID, "reason": "ETF inflows"}),
    "sentiment above range": json.dumps({**VALID, "sentiment_btc": 3}),
    "sentiment below range": json.dumps({**VALID, "relevance_sol": True, "sentiment_sol": -3}),
    "sentiment float": json.dumps({**VALID, "sentiment_btc": 1.5}),
    "sentiment bool": json.dumps({**VALID, "sentiment_btc": True}),
    "sentiment string": json.dumps({**VALID, "sentiment_btc": "1"}),
    "unknown event_type": json.dumps({**VALID, "event_type": "partnership"}),
    "event_type wrong case": json.dumps({**VALID, "event_type": "Regulation"}),
    "sentiment given when not relevant": json.dumps({**VALID, "sentiment_sol": 0}),
    "null sentiment when relevant": json.dumps({**VALID, "sentiment_btc": None}),
    "relevance as string": json.dumps({**VALID, "relevance_btc": "true"}),
    "confidence above 1": json.dumps({**VALID, "confidence": 1.2}),
    "confidence negative": json.dumps({**VALID, "confidence": -0.1}),
    "confidence string": json.dumps({**VALID, "confidence": "0.8"}),
    "confidence NaN": json.dumps(VALID).replace("0.8", "NaN"),
}


@pytest.mark.parametrize("case", INVALID)
def test_invalid_replies_are_rejected_and_not_stored(tmp_path, case, caplog):
    write_articles(tmp_path, ONE)
    with caplog.at_level(logging.WARNING):
        result, _ = run(tmp_path, [reply(INVALID[case])])
    assert (result["attempted"], result["labeled"], result["invalid"], result["failed"]) == (1, 0, 1, 0)
    assert read_csv(tmp_path / "news" / "labels.csv") == []
    assert "invalid reply for article a1" in caplog.text


def test_invalid_reply_does_not_stop_run_and_is_retried_later(tmp_path):
    write_articles(tmp_path, [("a1", "2026-10-24T10:00:00Z"), ("a2", "2026-10-24T11:00:00Z")])
    result, _ = run(tmp_path, [reply("not json"), reply(VALID)])
    assert (result["labeled"], result["invalid"]) == (1, 1)
    assert [r["article_id"] for r in read_csv(tmp_path / "news" / "labels.csv")] == ["a2"]

    result, session = run(tmp_path, [reply(VALID)])
    assert (result["attempted"], result["labeled"]) == (1, 1)
    assert "Title a1" in session.calls[0]["json"]["messages"][0]["content"]


# --- model unavailable ---

@pytest.mark.parametrize("failure", [
    FakeResponse(429, {"error": "rate limited"}),
    FakeResponse(500, {"error": "internal"}),
    FakeResponse(401, {"error": "unauthorized"}),
    requests.ConnectionError("connection refused"),
    requests.Timeout("timed out"),
    FakeResponse(200, {"error": "no choices"}),
])
def test_unavailable_model_stops_run_and_leaves_articles_unlabeled(tmp_path, failure):
    write_articles(tmp_path, [("a1", "2026-10-24T10:00:00Z"), ("a2", "2026-10-24T11:00:00Z"),
                              ("a3", "2026-10-24T12:00:00Z")])
    result, session = run(tmp_path, [reply(VALID), failure, reply(VALID)])

    assert len(session.calls) == 2
    assert (result["attempted"], result["labeled"], result["invalid"], result["failed"]) == (2, 1, 0, 1)
    assert result["error"] != ""
    assert [r["article_id"] for r in read_csv(tmp_path / "news" / "labels.csv")] == ["a1"]
    [logged] = read_csv(tmp_path / "news" / "label_runs.csv")
    assert list(logged) == RUN_COLUMNS
    assert (logged["attempted"], logged["labeled"], logged["failed"]) == ("2", "1", "1")

    result, _ = run(tmp_path, [reply(VALID), reply(VALID)])
    assert (result["attempted"], result["labeled"]) == (2, 2)
    assert [r["article_id"] for r in read_csv(tmp_path / "news" / "labels.csv")] == ["a1", "a2", "a3"]


def test_rate_limit_is_reported_as_such(tmp_path):
    write_articles(tmp_path, ONE)
    result, _ = run(tmp_path, [FakeResponse(429, {})])
    assert "429" in result["error"] and "rate limit" in result["error"]


def test_backoff_between_requests(tmp_path):
    write_articles(tmp_path, [("a1", "2026-10-24T10:00:00Z"), ("a2", "2026-10-24T11:00:00Z"),
                              ("a3", "2026-10-24T12:00:00Z")])
    pauses = []
    label_articles(KEY, tmp_path, session=FakeSession([reply(VALID)] * 3), now=fixed_now(),
                   sleep=pauses.append)
    assert pauses == [module.BACKOFF_S] * 2 and module.BACKOFF_S > 0


# --- ordering, skipping, append-only ---

def test_oldest_first_and_limit(tmp_path):
    write_articles(tmp_path, [("new", "2026-10-24T12:00:00Z"), ("old", "2026-10-23T08:00:00Z"),
                              ("mid", "2026-10-24T01:00:00Z")])
    result, _ = run(tmp_path, [reply(VALID), reply(VALID)], limit=2)
    assert result["attempted"] == 2
    assert [r["article_id"] for r in read_csv(tmp_path / "news" / "labels.csv")] == ["old", "mid"]


def test_already_labeled_articles_are_skipped_and_labels_append_only(tmp_path):
    write_articles(tmp_path, ONE)
    run(tmp_path, [reply(VALID)])
    path = tmp_path / "news" / "labels.csv"
    before = path.read_bytes()

    result, session = run(tmp_path, [])
    assert result["attempted"] == 0 and session.calls == []

    write_articles(tmp_path, ONE + [("a2", "2026-10-24T11:00:00Z")])
    run(tmp_path, [reply(VALID)])
    assert path.read_bytes().startswith(before)
    assert [r["article_id"] for r in read_csv(path)] == ["a1", "a2"]
    assert len(read_csv(tmp_path / "news" / "label_runs.csv")) == 3


def test_label_from_other_prompt_version_or_model_does_not_count(tmp_path):
    write_articles(tmp_path, [("a1", "2026-10-24T10:00:00Z"), ("a2", "2026-10-24T11:00:00Z")])
    path = tmp_path / "news" / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        base = {"labeled_utc": "2026-10-20T00:00:00Z", "model_reported": "x", "relevance_btc": "false",
                "relevance_sol": "false", "sentiment_btc": "", "sentiment_sol": "",
                "event_type": "other", "confidence": "0.5"}
        writer.writerow({**base, "article_id": "a1", "model_requested": MODEL, "prompt_version": "000000000000"})
        writer.writerow({**base, "article_id": "a2", "model_requested": "other-model",
                         "prompt_version": PROMPT_VERSION})
    result, _ = run(tmp_path, [reply(VALID), reply(VALID)])
    assert result["labeled"] == 2
    assert len(read_csv(path)) == 4


# --- what is sent: title and summary only ---

def test_request_contains_only_title_and_summary(tmp_path):
    """Leakage check: nothing but the article text reaches the model (no URL, source or times)."""
    write_articles(tmp_path, ONE)
    _, session = run(tmp_path, [reply(VALID)])
    payload = session.calls[0]["json"]
    assert set(payload) == {"model", "temperature", "messages"}
    assert payload["messages"] == build_messages("Title a1", "Summary a1")
    content = payload["messages"][0]["content"]
    for leaked in ("https://x.test/a1", "2026-10-24", "src", "a1\n"):
        assert leaked not in content.replace("Title a1", "").replace("Summary a1", "")


def test_prompt_forbids_outside_knowledge_and_lists_events():
    assert "Do not use outside knowledge" in PROMPT_TEMPLATE
    assert "after this article was written" in PROMPT_TEMPLATE
    assert "JSON only" in PROMPT_TEMPLATE
    assert all(event in PROMPT_TEMPLATE for event in EVENT_TYPES)


def test_braces_in_article_text_do_not_break_prompt():
    [message] = build_messages("Price {up}", "a {b} c")
    assert "Title: Price {up}" in message["content"] and "Summary: a {b} c" in message["content"]


# --- prompt version ---

def test_prompt_version_changes_when_template_changes():
    assert PROMPT_VERSION == prompt_version(PROMPT_TEMPLATE)
    assert len(PROMPT_VERSION) == 12 and int(PROMPT_VERSION, 16) >= 0
    assert prompt_version(PROMPT_TEMPLATE + " ") != PROMPT_VERSION
    assert prompt_version(PROMPT_TEMPLATE.replace("-2 to 2", "-3 to 3")) != PROMPT_VERSION


# --- the key ---

def test_read_env_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nOTHER=1\n\nexport FREELLMAPI_KEY=\"abc 123\"\n", encoding="utf-8")
    assert read_env_key(env) == "abc 123"
    env.write_text("FREELLMAPI_KEY=plain\n", encoding="utf-8")
    assert read_env_key(env) == "plain"


@pytest.mark.parametrize("content", [None, "OTHER=1\n", "FREELLMAPI_KEY=\n"])
def test_read_env_key_missing_raises_without_values(tmp_path, content):
    env = tmp_path / ".env"
    if content is not None:
        env.write_text(content, encoding="utf-8")
    with pytest.raises(KeyError) as info:
        read_env_key(env)
    assert "OTHER=1" not in str(info.value)


def test_key_is_sent_only_in_authorization_header(tmp_path):
    write_articles(tmp_path, ONE)
    _, session = run(tmp_path, [reply(VALID)])
    call = session.calls[0]
    assert call["headers"] == {"Authorization": f"Bearer {KEY}"}
    assert KEY not in json.dumps(call["json"])


def test_key_never_appears_in_files_or_logs(tmp_path, caplog, capsys):
    write_articles(tmp_path, [(f"a{i}", f"2026-10-24T1{i}:00:00Z") for i in range(5)])
    echo = {"error": f"bad key {KEY}", "model": KEY}
    responses = [
        reply(VALID, model=f"echo {KEY}"),
        reply(f"not json {KEY}"),
        reply({**VALID, "event_type": KEY}),
        reply(VALID, model="other-model"),
        FakeResponse(401, echo),
    ]
    with caplog.at_level(logging.DEBUG):
        run(tmp_path, responses)
        run(tmp_path, [requests.ConnectionError(f"failed with {KEY}")])

    for path in (tmp_path / "news").iterdir():
        assert KEY not in path.read_text(encoding="utf-8"), path.name
    assert KEY not in caplog.text
    captured = capsys.readouterr()
    assert KEY not in captured.out and KEY not in captured.err


# --- market fields ---

@pytest.mark.parametrize("label", [
    {**VALID, "relevance_market": False, "sentiment_market": None},
    {**VALID, "sentiment_market": 2},
    {**VALID, "sentiment_market": -2},
    {**VALID, "sentiment_market": 0},
    {**VALID, "relevance_btc": False, "sentiment_btc": None,
     "relevance_market": False, "sentiment_market": None, "event_type": "other"},
])
def test_valid_market_labels_are_accepted(label):
    assert parse_reply(json.dumps(label)) == (label, None)


MARKET_INVALID = {
    "market sentiment above range": {**VALID, "sentiment_market": 3},
    "market sentiment below range": {**VALID, "sentiment_market": -3},
    "market sentiment float": {**VALID, "sentiment_market": 0.5},
    "market sentiment string": {**VALID, "sentiment_market": "-1"},
    "market sentiment bool": {**VALID, "sentiment_market": False},
    "null market sentiment when relevant": {**VALID, "sentiment_market": None},
    "market sentiment given when not relevant": {**VALID, "relevance_market": False},
    "market relevance as string": {**VALID, "relevance_market": "false", "sentiment_market": None},
    "market relevance null": {**VALID, "relevance_market": None, "sentiment_market": None},
    "missing relevance_market": without("relevance_market"),
    "missing sentiment_market": without("sentiment_market"),
}


@pytest.mark.parametrize("case", MARKET_INVALID)
def test_invalid_market_fields_are_rejected_and_not_stored(tmp_path, case):
    assert parse_reply(json.dumps(MARKET_INVALID[case]))[0] is None
    write_articles(tmp_path, ONE)
    result, _ = run(tmp_path, [reply(MARKET_INVALID[case])])
    assert (result["labeled"], result["invalid"]) == (0, 1)
    assert read_csv(tmp_path / "news" / "labels.csv") == []


def test_market_fields_are_in_schema_and_prompt():
    assert {"relevance_market", "sentiment_market"} <= set(LABEL_COLUMNS)
    assert '"relevance_market"' in PROMPT_TEMPLATE and '"sentiment_market"' in PROMPT_TEMPLATE
    assert "crypto market overall" in PROMPT_TEMPLATE


# --- source boilerplate ---

BM_TITLE = "Bitcoin ETFs Lose Over $635M In One Day Following Oil Price Spike"
BM_BODY = f"{BM_TITLE} Investors this week were buying bitcoin ETFs but that all changed on Thursday."
BM_SUMMARY = (f"Bitcoin Magazine {BM_BODY} This post {BM_TITLE} first appeared on Bitcoin Magazine "
              "and is written by Mathew Di Salvo.")


def test_bitcoin_magazine_boilerplate_is_removed():
    assert strip_boilerplate("bitcoinmagazine", BM_SUMMARY) == BM_BODY
    trailing_only = f"{BM_BODY} This post {BM_TITLE} first appeared on Bitcoin Magazine and is written by J. Doe."
    assert strip_boilerplate("bitcoinmagazine", trailing_only) == BM_BODY
    assert strip_boilerplate("bitcoinmagazine", f"Bitcoin Magazine {BM_BODY}") == BM_BODY
    assert strip_boilerplate("bitcoinmagazine", BM_BODY) == BM_BODY


def test_bitcoin_magazine_text_that_is_not_boilerplate_is_kept():
    kept = "Bitcoin Magazines are popular. This post explains why. More text follows here."
    assert strip_boilerplate("bitcoinmagazine", kept) == kept
    assert strip_boilerplate("bitcoinmagazine", "") == ""


@pytest.mark.parametrize("source", ["coindesk", "cointelegraph", "decrypt", "theblock", "src"])
def test_other_sources_are_left_unchanged(source):
    assert strip_boilerplate(source, BM_SUMMARY) == BM_SUMMARY


def test_model_input_is_cleaned_but_stored_articles_are_not(tmp_path):
    path = tmp_path / "news" / "articles.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ARTICLE_COLUMNS)
        writer.writeheader()
        for article_id, source, published in [("bm", "bitcoinmagazine", "2026-10-24T10:00:00Z"),
                                              ("cd", "coindesk", "2026-10-24T11:00:00Z")]:
            writer.writerow({"id": article_id, "source": source, "title": BM_TITLE, "summary": BM_SUMMARY,
                             "url": f"https://x.test/{article_id}", "published_utc": published,
                             "fetched_utc": "2026-10-24T22:00:00Z"})
    before = path.read_bytes()

    _, session = run(tmp_path, [reply(VALID), reply(VALID)])

    sent_bm, sent_cd = (call["json"]["messages"] for call in session.calls)
    assert sent_bm == build_messages(BM_TITLE, BM_BODY)
    assert "first appeared on" not in sent_bm[0]["content"]
    assert "Summary: Bitcoin Magazine" not in sent_bm[0]["content"]
    assert sent_cd == build_messages(BM_TITLE, BM_SUMMARY)
    assert path.read_bytes() == before
