"""Label quality check: a stratified article sample for hand labeling, model
labels for that sample, and their comparison with the hand labels."""
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from cryptosense.news.collect import format_utc
from cryptosense.news.label import (
    ASSETS, BACKOFF_S, LABEL_COLUMNS, LABEL_FIELDS, MODEL, PROMPT_VERSION, TOKEN_FIELDS,
    ModelUnavailable,
    _append_row, _label_row, _read_csv, parse_reply, reported_model_error, request_label,
    strip_boilerplate,
)

MY_LABEL_COLUMNS = ["my_relevance_btc", "my_relevance_sol", "my_sentiment_btc",
                    "my_sentiment_sol", "my_event_type", "my_relevance_market", "my_sentiment_market"]
QC_SAMPLE_COLUMNS = ["article_id", "source", "title", "summary", *MY_LABEL_COLUMNS]


def draw_qc_sample(articles, per_source=5, seed=0, exclude=None):
    """Draw a random sample of ``per_source`` articles from every source.

    Articles are sorted by id before sampling, so the draw depends only on
    the set of articles, the excluded ids and the seed, not on the row order
    of the file. Sources are returned in alphabetical order.

    Parameters
    ----------
    articles : pandas.DataFrame
        The contents of ``articles.csv`` (needs ``id``, ``source``, ``title``
        and ``summary``).
    per_source : int
        Articles to draw from each source.
    seed : int
        Random seed.
    exclude : iterable of str or None
        Article ids that must not be drawn (for example an earlier sample).
        They are removed before sampling.

    Returns
    -------
    pandas.DataFrame
        Columns ``QC_SAMPLE_COLUMNS``, with the ``my_*`` columns empty.

    Raises
    ------
    ValueError
        If any source has fewer than ``per_source`` articles after exclusion.
    """
    if exclude is not None:
        articles = articles[~articles["id"].isin(set(exclude))]
    counts = articles["source"].value_counts()
    short = sorted(counts[counts < per_source].index)
    if short:
        raise ValueError(f"sources with fewer than {per_source} articles: {short}")
    ordered = articles.sort_values("id", kind="stable")
    sample = (ordered.groupby("source", sort=True, group_keys=False)
              .sample(n=per_source, random_state=seed)
              .rename(columns={"id": "article_id"}))
    for column in MY_LABEL_COLUMNS:
        sample[column] = ""
    return sample[QC_SAMPLE_COLUMNS].reset_index(drop=True)


def write_qc_sample(sample, path):
    """Write the sample to ``path``, refusing to overwrite an existing file.

    The file holds hand labels once filled in, so it is never replaced.

    Raises
    ------
    FileExistsError
        If ``path`` already exists.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; not overwriting hand labels")
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False, encoding="utf-8")


def add_missing_label_columns(path):
    """Add any ``MY_LABEL_COLUMNS`` missing from an existing sample file, empty.

    Rows, their order and every existing value (including hand labels) are
    kept; existing columns keep their positions and new columns are appended.
    A file that already has every column is left untouched.

    Returns
    -------
    list of str
        The columns that were added.
    """
    path = Path(path)
    sample = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in QC_SAMPLE_COLUMNS if column not in sample.columns]
    if missing:
        for column in missing:
            sample[column] = ""
        sample.to_csv(path, index=False, encoding="utf-8")
    return missing


# --- Stage 2: model labels for the sample and comparison with hand labels ---

QC_LABEL_COLUMNS = ["run", *LABEL_COLUMNS, "invalid_reason"]
RELEVANCE_ASSETS = ("market", "btc", "sol")
SENTIMENT_ASSETS = ("market", "btc")


def label_qc_sample(key, sample, path, runs=2, session=None, now=None, sleep=time.sleep,
                    backoff_s=BACKOFF_S):
    """Label every sample article ``runs`` times with the pinned model and prompt.

    Uses the same request, boilerplate stripping and validation as
    ``cryptosense.news.label``, but writes only to ``path`` (one row per run
    and article, with a ``run`` column), never to ``labels.csv``. An invalid
    reply is recorded with its reason and empty label fields, so it is not
    retried. Rows already in ``path`` for the same run, article, model and
    prompt version are skipped, so an interrupted labeling resumes. If the
    model is unavailable, or the API reports a model other than
    ``EXPECTED_REPORTED_MODEL``, nothing is stored for that article,
    labeling stops and the error is returned.

    Parameters
    ----------
    key : str
        The API key. Never logged or written to disk.
    sample : pandas.DataFrame
        The QC sample (needs ``article_id``, ``source``, ``title``, ``summary``).
    path : Path
        The output CSV (``qc_labels.csv``).
    runs : int
        Number of labeling runs over the whole sample.
    session, now, sleep
        As in ``label_articles``.
    backoff_s : float
        Seconds between requests. Defaults to the production ``BACKOFF_S``.

    Returns
    -------
    dict
        Counts ``attempted``, ``labeled``, ``invalid``, ``failed`` and ``error``,
        and the totals of ``TOKEN_FIELDS`` over every response received in
        this call (including invalid and rejected replies).

    Raises
    ------
    ValueError
        If ``path`` is named ``labels.csv``.
    """
    path = Path(path)
    if path.name == "labels.csv":
        raise ValueError("QC labels must not be written to labels.csv")
    now = now or (lambda: datetime.now(timezone.utc))
    session = session or requests.Session()
    done = {(row["run"], row["article_id"]) for row in _read_csv(path)
            if row["model_requested"] == MODEL and row["prompt_version"] == PROMPT_VERSION}

    result = {"attempted": 0, "labeled": 0, "invalid": 0, "failed": 0, "error": "",
              **{field: 0 for field in TOKEN_FIELDS}}
    for run in range(1, runs + 1):
        for article in sample.to_dict("records"):
            if (str(run), article["article_id"]) in done:
                continue
            if result["attempted"]:
                sleep(backoff_s)
            result["attempted"] += 1
            try:
                summary = strip_boilerplate(article["source"], article["summary"])
                content, model_reported, usage = request_label(session, key, article["title"], summary)
            except ModelUnavailable as exc:
                result["failed"] += 1
                result["error"] = str(exc)
                return result
            for field in TOKEN_FIELDS:
                result[field] += usage[field]
            mismatch = reported_model_error(model_reported)
            if mismatch:
                result["failed"] += 1
                result["error"] = mismatch
                return result
            label, reason = parse_reply(content)
            if label is None:
                row = {"article_id": article["article_id"], "model_requested": MODEL,
                       "model_reported": model_reported, "prompt_version": PROMPT_VERSION,
                       "labeled_utc": format_utc(now()), **{field: "" for field in LABEL_FIELDS}}
                result["invalid"] += 1
            else:
                row = _label_row(article["article_id"], model_reported, format_utc(now()), label)
                reason = ""
                result["labeled"] += 1
            _append_row(path, QC_LABEL_COLUMNS, {"run": run, **row, "invalid_reason": reason})
    return result


def _to_bool(value):
    """Parse a CSV relevance cell ("true"/"True"/"false"/"False") to a bool."""
    text = str(value).strip().lower()
    if text not in ("true", "false"):
        raise ValueError(f"not a true/false value: {value!r}")
    return text == "true"


def _to_sentiment(value):
    """Parse a CSV sentiment cell to an int, or None for an empty cell."""
    text = str(value).strip()
    return None if text == "" else int(float(text))


def _direction(sentiment):
    """Map a sentiment score to -1 (negative), 0 (neutral) or 1 (positive)."""
    return (sentiment > 0) - (sentiment < 0)


def _share(matches):
    """Return n, matches, the share and its standard error sqrt(p(1-p)/n)."""
    n = len(matches)
    agree = int(sum(matches))
    if not n:
        return {"n": 0, "agree": 0, "share": float("nan"), "se": float("nan")}
    share = agree / n
    return {"n": n, "agree": agree, "share": share, "se": math.sqrt(share * (1 - share) / n)}


def _chance(hand, model):
    """Agreement expected by chance from the two sets of marginal frequencies."""
    hand, model = pd.Series(list(hand), dtype=object), pd.Series(list(model), dtype=object)
    if hand.empty:
        return float("nan")
    p_hand = hand.value_counts(normalize=True)
    p_model = model.value_counts(normalize=True)
    return float(sum(p_hand.get(value, 0) * p_model.get(value, 0)
                     for value in set(p_hand.index) | set(p_model.index)))


def split_hand_labeled(sample):
    """Split a QC sample into hand-labeled articles and articles with no hand label.

    An article counts as unlabeled when every ``MY_LABEL_COLUMNS`` cell is
    empty (or whitespace). Unlabeled articles are left out of the
    comparison with hand labels but can still be used for run-to-run
    consistency.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(labeled, excluded)``, each keeping the original row order.
    """
    empty = sample[MY_LABEL_COLUMNS].apply(lambda column: column.astype(str).str.strip() == "")
    unlabeled = empty.all(axis=1)
    return sample[~unlabeled], sample[unlabeled]


def current_model_labels(labels):
    """Keep only valid QC label rows from the pinned ``MODEL`` and current ``PROMPT_VERSION``.

    Rows from earlier models or prompts stay in the file (append-only) but
    are never compared.
    """
    keep = ((labels["model_requested"] == MODEL) & (labels["prompt_version"] == PROMPT_VERSION)
            & (labels["invalid_reason"] == ""))
    return labels[keep]


def merge_hand_and_model(sample, labels, run=1):
    """Join hand labels with the valid model labels of one run.

    Only rows kept by ``current_model_labels`` are used. Returns one row per
    sample article that has such a label in ``run``, with parsed columns
    ``hand_*`` and ``model_*``: relevance as bool, sentiment as int or None,
    event type as str. An empty hand sentiment cell is treated as null.
    """
    labels = current_model_labels(labels)
    model = labels[labels["run"].astype(str) == str(run)]
    merged = sample.merge(model, on="article_id", how="inner", validate="one_to_one")
    out = merged[["article_id", "source", "title", "summary"]].copy()
    for asset in ASSETS:
        out[f"hand_relevance_{asset}"] = merged[f"my_relevance_{asset}"].map(_to_bool)
        out[f"model_relevance_{asset}"] = merged[f"relevance_{asset}"].map(_to_bool)
        for prefix, column in (("hand", f"my_sentiment_{asset}"), ("model", f"sentiment_{asset}")):
            out[f"{prefix}_sentiment_{asset}"] = pd.Series(
                [_to_sentiment(value) for value in merged[column]], index=merged.index, dtype=object)
    out["hand_event_type"] = merged["my_event_type"].str.strip()
    out["model_event_type"] = merged["event_type"]
    return out.reset_index(drop=True)


def _both_scored(merged, asset):
    """Rows where hand and model both marked ``asset`` relevant and both gave a sentiment."""
    both = merged[merged[f"hand_relevance_{asset}"] & merged[f"model_relevance_{asset}"]]
    return both[both[f"hand_sentiment_{asset}"].notna() & both[f"model_sentiment_{asset}"].notna()]


def agreement(merged):
    """Agreement between hand and model labels on a ``merge_hand_and_model`` frame.

    Returns
    -------
    pandas.DataFrame
        One row per measure: relevance for market, BTC and SOL; for market
        and BTC, on articles both marked relevant, sentiment direction
        (negative / neutral / positive) agreement and the share within one
        step; and event type. Columns ``n``, ``agree``, ``share``, ``se`` and
        ``chance`` (agreement expected from the two marginal distributions;
        NaN for the within-one-step measure).
    """
    rows = {}
    for asset in RELEVANCE_ASSETS:
        hand, model = merged[f"hand_relevance_{asset}"], merged[f"model_relevance_{asset}"]
        rows[f"relevance_{asset}"] = {**_share(list(hand == model)), "chance": _chance(hand, model)}
    for asset in SENTIMENT_ASSETS:
        both = _both_scored(merged, asset)
        hand_scores = list(both[f"hand_sentiment_{asset}"])
        model_scores = list(both[f"model_sentiment_{asset}"])
        hand = [_direction(s) for s in hand_scores]
        model = [_direction(s) for s in model_scores]
        rows[f"sentiment_direction_{asset}"] = {**_share([h == m for h, m in zip(hand, model)]),
                                                "chance": _chance(hand, model)}
        rows[f"sentiment_within_one_{asset}"] = {
            **_share([abs(h - m) <= 1 for h, m in zip(hand_scores, model_scores)]),
            "chance": float("nan")}
    hand, model = merged["hand_event_type"], merged["model_event_type"]
    rows["event_type"] = {**_share(list(hand == model)), "chance": _chance(hand, model)}
    return pd.DataFrame.from_dict(rows, orient="index")


def disagreements(merged):
    """List every hand-versus-model disagreement on a ``merge_hand_and_model`` frame.

    Covered: relevance for market, BTC and SOL; sentiment for market and BTC
    where both marked the asset relevant and the scores differ; event type.

    Returns
    -------
    pandas.DataFrame
        Columns ``field``, ``source``, ``title``, ``summary``, ``hand``, ``model``.
    """
    columns = ["field", "source", "title", "summary", "hand", "model"]
    rows = []
    for record in merged.to_dict("records"):
        found = []
        for asset in RELEVANCE_ASSETS:
            if record[f"hand_relevance_{asset}"] != record[f"model_relevance_{asset}"]:
                found.append((f"relevance_{asset}", record[f"hand_relevance_{asset}"],
                              record[f"model_relevance_{asset}"]))
        for asset in SENTIMENT_ASSETS:
            hand, model = record[f"hand_sentiment_{asset}"], record[f"model_sentiment_{asset}"]
            if (record[f"hand_relevance_{asset}"] and record[f"model_relevance_{asset}"]
                    and hand is not None and model is not None and hand != model):
                found.append((f"sentiment_{asset}", hand, model))
        if record["hand_event_type"] != record["model_event_type"]:
            found.append(("event_type", record["hand_event_type"], record["model_event_type"]))
        rows += [dict(zip(columns, (field, record["source"], record["title"], record["summary"],
                                    hand, model)))
                 for field, hand, model in found]
    return pd.DataFrame(rows, columns=columns)


def run_consistency(labels, first=1, second=2):
    """Share of articles with an identical value in two runs, per label field.

    Only rows kept by ``current_model_labels`` are used, and only articles
    with such a label in both runs count; values are compared as stored in
    the CSV. For the sentiment fields, the share with the same direction
    (negative / neutral / positive) is reported alongside; a null sentiment
    is its own category, so null versus 0 counts as different.

    Returns
    -------
    pandas.DataFrame
        Indexed by field, columns ``n``, ``agree``, ``share`` and ``se`` for
        exact identity, and ``direction_agree``, ``direction_share`` and
        ``direction_se`` (NaN for fields that are not sentiments).
    """
    valid = current_model_labels(labels)
    a = valid[valid["run"].astype(str) == str(first)].set_index("article_id")
    b = valid[valid["run"].astype(str) == str(second)].set_index("article_id")
    common = a.index.intersection(b.index)
    rows = {}
    for field in LABEL_FIELDS:
        rows[field] = _share(list(a.loc[common, field] == b.loc[common, field]))
        if field.startswith("sentiment_"):
            same = [_stored_direction(x) == _stored_direction(y)
                    for x, y in zip(a.loc[common, field], b.loc[common, field])]
            direction = _share(same)
            rows[field].update(direction_agree=direction["agree"], direction_share=direction["share"],
                               direction_se=direction["se"])
        else:
            rows[field].update(direction_agree=float("nan"), direction_share=float("nan"),
                               direction_se=float("nan"))
    return pd.DataFrame.from_dict(rows, orient="index")


def _stored_direction(value):
    """Direction of a stored sentiment cell: -1, 0 or 1, or None for an empty cell."""
    sentiment = _to_sentiment(value)
    return None if sentiment is None else _direction(sentiment)
