import json
from datetime import datetime

import pandas as pd
import pytest

from cryptosense.news.label import (
    EXPECTED_REPORTED_MODEL, LABEL_FIELDS, MODEL, PROMPT_VERSION, build_messages,
)
from cryptosense.news.qc import (
    MY_LABEL_COLUMNS, QC_LABEL_COLUMNS, QC_SAMPLE_COLUMNS, add_missing_label_columns, agreement,
    disagreements, draw_qc_sample, label_qc_sample, merge_hand_and_model, run_consistency,
    write_qc_sample,
)


def make_articles(counts):
    """Articles with ``counts[source]`` rows per source."""
    rows = [{"id": f"{source}-{i:03d}", "source": source, "title": f"T {source} {i}",
             "summary": f"S {source} {i}", "url": "u", "published_utc": "2026-10-24T00:00:00Z",
             "fetched_utc": "2026-10-24T00:00:00Z"}
            for source, n in counts.items() for i in range(n)]
    return pd.DataFrame(rows)


COUNTS = {"b": 12, "a": 30, "c": 5}


def test_sample_has_n_per_source_and_empty_label_columns():
    sample = draw_qc_sample(make_articles(COUNTS), per_source=5, seed=0)
    assert list(sample.columns) == QC_SAMPLE_COLUMNS
    assert sample["source"].value_counts().to_dict() == {"a": 5, "b": 5, "c": 5}
    assert list(sample["source"]) == ["a"] * 5 + ["b"] * 5 + ["c"] * 5
    assert sample["article_id"].is_unique
    assert (sample[MY_LABEL_COLUMNS] == "").all().all()


def test_sample_rows_match_articles():
    articles = make_articles(COUNTS)
    sample = draw_qc_sample(articles)
    merged = sample.merge(articles, left_on="article_id", right_on="id", suffixes=("", "_a"))
    assert len(merged) == len(sample)
    assert (merged["title"] == merged["title_a"]).all()
    assert (merged["summary"] == merged["summary_a"]).all()
    assert (merged["source"] == merged["source_a"]).all()


def test_sample_is_reproducible_and_independent_of_row_order():
    articles = make_articles(COUNTS)
    first = draw_qc_sample(articles, seed=0)
    assert first.equals(draw_qc_sample(articles, seed=0))
    shuffled = articles.sample(frac=1, random_state=42)
    assert first.equals(draw_qc_sample(shuffled, seed=0))
    assert not first.equals(draw_qc_sample(articles, seed=1))


def test_source_with_too_few_articles_raises():
    with pytest.raises(ValueError, match="'c'"):
        draw_qc_sample(make_articles({"a": 10, "c": 4}), per_source=5)


def test_write_refuses_to_overwrite(tmp_path):
    path = tmp_path / "news" / "qc_sample.csv"
    sample = draw_qc_sample(make_articles(COUNTS))
    write_qc_sample(sample, path)
    back = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert back.equals(sample)

    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_qc_sample(sample.iloc[:1], path)
    assert path.read_bytes() == before


def test_add_missing_label_columns_keeps_rows_and_hand_labels(tmp_path):
    path = tmp_path / "qc_sample.csv"
    old_columns = [c for c in QC_SAMPLE_COLUMNS if c not in ("my_relevance_market", "my_sentiment_market")]
    old = draw_qc_sample(make_articles(COUNTS))[old_columns]
    old.loc[0, "my_relevance_btc"] = "true"
    old.loc[3, "my_event_type"] = "regulation"
    old.to_csv(path, index=False, encoding="utf-8")

    assert add_missing_label_columns(path) == ["my_relevance_market", "my_sentiment_market"]
    new = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(new.columns) == old_columns + ["my_relevance_market", "my_sentiment_market"]
    assert new[old_columns].equals(old)
    assert (new[["my_relevance_market", "my_sentiment_market"]] == "").all().all()

    before = path.read_bytes()
    assert add_missing_label_columns(path) == []
    assert path.read_bytes() == before


def test_excluded_articles_are_never_drawn_and_draw_is_reproducible():
    articles = make_articles({"b": 12, "a": 30, "c": 10})
    first = draw_qc_sample(articles, per_source=5, seed=0)
    second = draw_qc_sample(articles, per_source=5, seed=1, exclude=first["article_id"])
    assert not set(second["article_id"]) & set(first["article_id"])
    assert second["source"].value_counts().to_dict() == {"a": 5, "b": 5, "c": 5}
    c_ids = set(articles.loc[articles["source"] == "c", "id"])
    assert set(second.loc[second["source"] == "c", "article_id"]) == c_ids - set(first["article_id"])
    assert second.equals(draw_qc_sample(articles.sample(frac=1, random_state=3), per_source=5,
                                        seed=1, exclude=list(first["article_id"])))
    assert list(second.columns) == QC_SAMPLE_COLUMNS
    assert (second[MY_LABEL_COLUMNS] == "").all().all()


def test_exclusion_that_leaves_too_few_articles_raises():
    articles = make_articles({"a": 10, "c": 8})
    first = draw_qc_sample(articles, per_source=5, seed=0)
    with pytest.raises(ValueError, match="'c'"):
        draw_qc_sample(articles, per_source=5, seed=1, exclude=first["article_id"])


def test_market_columns_are_hand_label_columns():
    assert MY_LABEL_COLUMNS[-2:] == ["my_relevance_market", "my_sentiment_market"]


# --- Stage 2: model labels for the sample ---

KEY = "sk-test-SECRET-key-0123456789"
VALID = {"relevance_btc": True, "relevance_sol": False, "relevance_market": True,
         "sentiment_btc": 1, "sentiment_sol": None, "sentiment_market": -1,
         "event_type": "institutional", "confidence": 0.8}


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


def reply(content):
    if not isinstance(content, str):
        content = json.dumps(content)
    return FakeResponse(200, {"model": EXPECTED_REPORTED_MODEL,
                              "choices": [{"message": {"content": content}}]})


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.responses.pop(0)


def small_sample():
    summary = ("Bitcoin Magazine Body one. This post T1 first appeared on Bitcoin Magazine "
               "and is written by X.")
    return pd.DataFrame([
        {"article_id": "a1", "source": "bitcoinmagazine", "title": "T1", "summary": summary},
        {"article_id": "a2", "source": "coindesk", "title": "T2", "summary": ""},
    ])


def label(path, responses):
    session = FakeSession(responses)
    result = label_qc_sample(KEY, small_sample(), path, session=session, sleep=lambda s: None,
                             now=lambda: datetime.fromisoformat("2026-10-25T09:00:00+00:00"))
    return result, session


def test_qc_labels_two_runs_written_only_to_given_path(tmp_path):
    path = tmp_path / "news" / "qc_labels.csv"
    result, session = label(path, [reply(VALID)] * 4)
    assert (result["attempted"], result["labeled"], result["invalid"], result["failed"]) == (4, 4, 0, 0)
    assert sorted(p.name for p in path.parent.iterdir()) == ["qc_labels.csv"]
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(rows.columns) == QC_LABEL_COLUMNS
    assert list(zip(rows["run"], rows["article_id"])) == [("1", "a1"), ("1", "a2"),
                                                          ("2", "a1"), ("2", "a2")]
    assert (rows["model_requested"] == MODEL).all()
    assert (rows["prompt_version"] == PROMPT_VERSION).all()
    assert (rows["invalid_reason"] == "").all()
    assert rows.loc[0, "sentiment_market"] == "-1" and rows.loc[0, "sentiment_sol"] == ""


def test_qc_request_is_title_and_cleaned_summary_only(tmp_path):
    """Leakage check: only the title and boilerplate-stripped summary reach the model."""
    _, session = label(tmp_path / "qc_labels.csv", [reply(VALID)] * 4)
    payload = session.calls[0]["json"]
    assert set(payload) == {"model", "temperature", "messages"}
    assert payload["model"] == MODEL and payload["temperature"] == 0
    assert payload["messages"] == build_messages("T1", "Body one.")
    assert session.calls[0]["headers"] == {"Authorization": f"Bearer {KEY}"}
    content = payload["messages"][0]["content"].replace("T1", "").replace("Body one.", "")
    for leaked in ("a1", "bitcoinmagazine", KEY):
        assert leaked not in content


def test_qc_backoff_defaults_to_production_and_can_be_set(tmp_path):
    from cryptosense.news.label import BACKOFF_S
    for kwargs, expected in (({}, BACKOFF_S), ({"backoff_s": 10}, 10)):
        pauses = []
        label_qc_sample(KEY, small_sample(), tmp_path / f"qc{len(kwargs)}.csv",
                        session=FakeSession([reply(VALID)] * 4), sleep=pauses.append,
                        now=lambda: datetime.fromisoformat("2026-10-25T09:00:00+00:00"), **kwargs)
        assert pauses == [expected] * 3
    assert BACKOFF_S == 12.0


def test_qc_mismatched_reported_model_is_not_stored_and_stops(tmp_path):
    path = tmp_path / "qc_labels.csv"
    other = FakeResponse(200, {"model": "gemini-3.8-flash",
                               "choices": [{"message": {"content": json.dumps(VALID)}}]})
    result, session = label(path, [reply(VALID), other, reply(VALID)])
    assert len(session.calls) == 2
    assert (result["attempted"], result["labeled"], result["failed"]) == (2, 1, 1)
    assert "mismatch" in result["error"]
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(rows["article_id"]) == ["a1"]
    assert (rows["model_reported"] == EXPECTED_REPORTED_MODEL).all()


def test_qc_refuses_labels_csv(tmp_path):
    with pytest.raises(ValueError):
        label(tmp_path / "labels.csv", [])
    assert not (tmp_path / "labels.csv").exists()


def test_qc_invalid_reply_recorded_and_unavailable_stops_then_resumes(tmp_path):
    path = tmp_path / "qc_labels.csv"
    result, _ = label(path, [reply("not json"), reply(VALID), FakeResponse(429, {})])
    assert (result["attempted"], result["labeled"], result["invalid"], result["failed"]) == (3, 1, 1, 1)
    assert "429" in result["error"]
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(rows["article_id"]) == ["a1", "a2"]
    assert rows.loc[0, "invalid_reason"] == "reply is not valid JSON"
    assert (rows.loc[0, LABEL_FIELDS] == "").all()

    before = path.read_bytes()
    result, session = label(path, [reply(VALID), reply(VALID)])
    assert result["attempted"] == 2 and len(session.calls) == 2
    assert path.read_bytes().startswith(before)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(zip(rows["run"], rows["article_id"])) == [("1", "a1"), ("1", "a2"),
                                                          ("2", "a1"), ("2", "a2")]
    assert KEY not in path.read_text(encoding="utf-8")


# --- Stage 2: comparison with hand labels ---

def hand_row(article_id, rel_btc, rel_mkt, s_btc, s_mkt, event):
    return {"article_id": article_id, "source": "src", "title": f"T {article_id}",
            "summary": f"S {article_id}", "my_relevance_btc": rel_btc, "my_relevance_sol": "False",
            "my_sentiment_btc": s_btc, "my_sentiment_sol": "", "my_event_type": event,
            "my_relevance_market": rel_mkt, "my_sentiment_market": s_mkt}


def model_row(run, article_id, rel_btc, rel_mkt, s_btc, s_mkt, event, invalid=""):
    return {"run": str(run), "article_id": article_id, "model_requested": MODEL,
            "model_reported": MODEL, "prompt_version": PROMPT_VERSION, "labeled_utc": "x",
            "relevance_btc": rel_btc, "relevance_sol": "false", "relevance_market": rel_mkt,
            "sentiment_btc": s_btc, "sentiment_sol": "", "sentiment_market": s_mkt,
            "event_type": event, "confidence": "0.9", "invalid_reason": invalid}


HAND = pd.DataFrame([
    hand_row("a", "True", "True", "1", "1", "macro"),
    hand_row("b", "True", "False", "0", "", "legal"),
    hand_row("c", "False", "True", "", "-1", "other"),
    hand_row("d", "True", "True", "-1", "0", "security"),
])
MODEL_LABELS = pd.DataFrame([
    model_row(1, "a", "true", "true", "2", "1", "macro"),
    model_row(1, "b", "true", "true", "-1", "0", "legal"),
    model_row(1, "c", "false", "true", "", "-2", "regulation"),
    model_row(1, "d", "", "", "", "", "", invalid="reply is not valid JSON"),
    model_row(2, "a", "true", "true", "2", "1", "macro"),
    model_row(2, "b", "true", "true", "-1", "1", "legal"),
    model_row(2, "c", "false", "true", "", "-2", "other"),
])


def test_merge_treats_empty_hand_sentiment_as_null_and_drops_invalid():
    merged = merge_hand_and_model(HAND, MODEL_LABELS, run=1)
    assert list(merged["article_id"]) == ["a", "b", "c"]
    assert merged.loc[1, "hand_sentiment_market"] is None
    assert merged.loc[0, "hand_relevance_btc"] and not merged.loc[2, "model_relevance_btc"]
    assert merged.loc[0, "model_sentiment_btc"] == 2


def test_agreement_figures():
    table = agreement(merge_hand_and_model(HAND, MODEL_LABELS, run=1))
    assert table.loc["relevance_market", ["n", "agree"]].tolist() == [3, 2]
    assert table.loc["relevance_btc", ["n", "agree"]].tolist() == [3, 3]
    assert table.loc["relevance_sol", ["n", "agree"]].tolist() == [3, 3]
    assert table.loc["sentiment_direction_market", ["n", "agree"]].tolist() == [2, 2]
    assert table.loc["sentiment_within_one_market", ["n", "agree"]].tolist() == [2, 2]
    assert table.loc["sentiment_direction_btc", ["n", "agree"]].tolist() == [2, 1]
    assert table.loc["sentiment_within_one_btc", ["n", "agree"]].tolist() == [2, 2]
    assert table.loc["event_type", ["n", "agree"]].tolist() == [3, 2]
    assert table.loc["event_type", "share"] == pytest.approx(2 / 3)
    assert table.loc["event_type", "se"] == pytest.approx(((2 / 3) * (1 / 3) / 3) ** 0.5)
    assert table.loc["relevance_sol", "chance"] == pytest.approx(1.0)
    assert table.loc["sentiment_direction_btc", "chance"] == pytest.approx(0.25)


def test_disagreements_lists_each_field():
    found = disagreements(merge_hand_and_model(HAND, MODEL_LABELS, run=1))
    assert list(zip(found["title"], found["field"], found["hand"], found["model"])) == [
        ("T a", "sentiment_btc", 1, 2),
        ("T b", "relevance_market", False, True),
        ("T b", "sentiment_btc", 0, -1),
        ("T c", "sentiment_market", -1, -2),
        ("T c", "event_type", "other", "regulation"),
    ]
    assert found.loc[0, "summary"] == "S a"


def test_rows_from_other_models_or_prompts_are_not_compared():
    old_model = {**model_row(1, "a", "false", "false", "", "", "other"),
                 "model_requested": "gemini-3.8-flash", "model_reported": "gemini-3.8-flash"}
    old_prompt = {**model_row(2, "b", "false", "false", "", "", "other"), "prompt_version": "000000000000"}
    labels = pd.concat([pd.DataFrame([old_model, old_prompt]), MODEL_LABELS], ignore_index=True)
    merged = merge_hand_and_model(HAND, labels, run=1)
    assert merged.equals(merge_hand_and_model(HAND, MODEL_LABELS, run=1))
    assert run_consistency(labels).equals(run_consistency(MODEL_LABELS))


def test_run_consistency_reports_sentiment_direction_alongside_identity():
    table = run_consistency(MODEL_LABELS)
    # market: a 1/1, b 0/1, c -2/-2 -> identical 2 of 3; direction 0 vs + differs -> 2 of 3
    assert table.loc["sentiment_market", ["direction_agree", "direction_share"]].tolist() == [2, 2 / 3]
    # btc: a 2/2, b -1/-1, c null/null -> identical 3, direction 3
    assert table.loc["sentiment_btc", "direction_agree"] == 3
    assert pd.isna(table.loc["event_type", "direction_share"])

    labels = pd.DataFrame([model_row(1, "x", "true", "true", "1", "", "macro"),
                           model_row(2, "x", "true", "true", "2", "0", "macro"),
                           model_row(1, "y", "true", "true", "-1", "0", "macro"),
                           model_row(2, "y", "true", "true", "-2", "0", "macro")])
    labels.loc[0, "relevance_market"] = "false"
    table = run_consistency(labels)
    assert table.loc["sentiment_btc", ["agree", "direction_agree"]].tolist() == [0, 2]
    assert table.loc["sentiment_market", ["agree", "direction_agree"]].tolist() == [1, 1]
    assert table.loc["sentiment_btc", "direction_se"] == 0


def test_run_consistency_per_field():
    table = run_consistency(MODEL_LABELS)
    assert list(table.index) == LABEL_FIELDS
    assert table.loc["sentiment_market", ["n", "agree"]].tolist() == [3, 2]
    assert table.loc["event_type", ["n", "agree"]].tolist() == [3, 2]
    assert table.loc["relevance_btc", ["n", "agree"]].tolist() == [3, 3]
