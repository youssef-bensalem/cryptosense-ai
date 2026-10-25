"""Label collected news articles with an LLM via a local OpenAI-compatible API.

Usage: ``python -m cryptosense.news.label [--limit N] [--data-dir PATH]``

Each article's title and summary (nothing else) are sent to one pinned model
at temperature 0. A reply is stored only if it is JSON that matches the label
schema exactly; anything else leaves the article unlabeled for a later run.
Labels are appended to ``labels.csv`` and one row per run to
``label_runs.csv``. The API key is read from ``.env`` and is never printed,
logged or stored.
"""
import argparse
import csv
import hashlib
import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from cryptosense.news.collect import DEFAULT_DATA_DIR, format_utc

logger = logging.getLogger(__name__)

ENDPOINT = "http://localhost:31415/v1/chat/completions"
MODEL = "gpt-oss-120b"
# The ``model`` field the API must report for MODEL. Any other value means a
# substituted model: the reply is not stored and the run stops.
EXPECTED_REPORTED_MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0
TIMEOUT_S = 60
BACKOFF_S = 12.0
ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
KEY_NAME = "FREELLMAPI_KEY"

EVENT_TYPES = (
    "regulation", "security", "institutional", "macro", "exchange",
    "technology", "adoption", "legal", "market_commentary", "other",
)
ASSETS = ("btc", "sol", "market")
LABEL_FIELDS = [
    "relevance_btc", "relevance_sol", "relevance_market",
    "sentiment_btc", "sentiment_sol", "sentiment_market", "event_type", "confidence",
]
LABEL_COLUMNS = ["article_id", "model_requested", "model_reported", "prompt_version", "labeled_utc",
                 *LABEL_FIELDS]
TOKEN_FIELDS = ["prompt_tokens", "completion_tokens", "total_tokens"]
RUN_COLUMNS = ["run_utc", "attempted", "labeled", "invalid", "failed", "error", *TOKEN_FIELDS]

PROMPT_TEMPLATE = """You label crypto news articles for a research dataset.

Judge ONLY the text given below. Do not use outside knowledge, and in particular
do not use any knowledge of events or prices after this article was written.

How to judge sentiment:
- Sentiment judges the implication of a NEW event for prices. A report that the
  price moved, without a new underlying event, is 0.
- +2 or -2 only for major, rare events: a country-level ban or approval, a major
  exchange collapse or hack, record institutional moves.
- +1 or -1 for ordinary positive or negative news.
- 0 for neutral, mixed or purely factual news, and for price-move reports.

Return JSON only: a single JSON object, with no code fences, no comments and no
other text. It must have exactly these keys:
- "relevance_btc": true or false. Is the article materially about Bitcoin (BTC)?
- "relevance_sol": true or false. Is the article materially about Solana (SOL)?
- "relevance_market": true or false. true if the article is about crypto assets,
  crypto markets or the crypto industry, including news about a single coin such
  as Bitcoin. false only for articles not about crypto (for example general
  technology or security news).
- "sentiment_btc": integer from -2 to 2 for the article's implication for BTC,
  using the scale above, or null if relevance_btc is false.
- "sentiment_sol": integer from -2 to 2 for the article's implication for SOL,
  using the scale above, or null if relevance_sol is false.
- "sentiment_market": integer from -2 to 2 for the article's implication for the
  crypto market overall, using the scale above, or null if relevance_market is false.
- "event_type": exactly one of: regulation, security, institutional, macro,
  exchange, technology, adoption, legal, market_commentary, other.
- "confidence": number from 0 to 1, your confidence in this label.

Title: {title}
Summary: {summary}
"""


def prompt_version(template):
    """Return the first 12 hex characters of the SHA-256 of a prompt template."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]


PROMPT_VERSION = prompt_version(PROMPT_TEMPLATE)


def read_env_key(path=ENV_PATH, name=KEY_NAME):
    """Read one variable from a ``.env`` file without touching the environment.

    Supports ``NAME=value`` lines, optional ``export`` prefixes, surrounding
    quotes, blank lines and ``#`` comments.

    Raises
    ------
    KeyError
        If the file does not exist or does not define ``name``. The message
        never contains a value from the file.
    """
    path = Path(path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1]
                if value:
                    return value
    raise KeyError(f"{name} is not set in {path}")


BOILERPLATE_SOURCE = "bitcoinmagazine"
LEADING_BOILERPLATE = "Bitcoin Magazine"
TRAILING_BOILERPLATE = re.compile(r"This post .+ first appeared on .+ and is written by .+", re.DOTALL)


def strip_boilerplate(source, summary):
    """Remove source boilerplate from a summary before it is sent to the model.

    For Bitcoin Magazine only: drops a leading "Bitcoin Magazine" and a final
    "This post ... first appeared on ... and is written by ..." sentence.
    Summaries from other sources are returned unchanged. Stored articles are
    never modified; this applies to the model input only.
    """
    if source != BOILERPLATE_SOURCE:
        return summary
    text = summary
    if text == LEADING_BOILERPLATE or text.startswith(LEADING_BOILERPLATE + " "):
        text = text[len(LEADING_BOILERPLATE):].lstrip()
    start = text.rfind("This post ")
    if start >= 0 and TRAILING_BOILERPLATE.fullmatch(text[start:].rstrip()):
        text = text[:start].rstrip()
    return text


def build_messages(title, summary, template=PROMPT_TEMPLATE):
    """Build the chat messages for one article from its title and summary only."""
    return [{"role": "user", "content": template.format(title=title, summary=summary)}]


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_label(label):
    """Check a parsed reply against the label schema.

    The reply must be a JSON object with exactly the keys in ``LABEL_FIELDS``:
    booleans for the relevance fields; an integer from -2 to 2 for each
    sentiment, null exactly when the matching relevance is false; an
    ``event_type`` from ``EVENT_TYPES``; and a ``confidence`` from 0 to 1.

    Returns
    -------
    str or None
        None if the label is valid, otherwise the reason it is not. Reasons
        never quote values from the reply, so they are safe to log.
    """
    if not isinstance(label, dict):
        return "reply is not a JSON object"
    if set(label) != set(LABEL_FIELDS):
        missing = sorted(set(LABEL_FIELDS) - set(label))
        extra = sorted(set(label) - set(LABEL_FIELDS))
        return f"wrong keys (missing {missing}, extra {extra})"
    for asset in ASSETS:
        relevant = label[f"relevance_{asset}"]
        sentiment = label[f"sentiment_{asset}"]
        if not isinstance(relevant, bool):
            return f"relevance_{asset} is not true/false"
        if not relevant and sentiment is not None:
            return f"sentiment_{asset} given although relevance_{asset} is false"
        if relevant and sentiment is None:
            return f"sentiment_{asset} is null although relevance_{asset} is true"
        if relevant and not (_is_int(sentiment) and -2 <= sentiment <= 2):
            return f"sentiment_{asset} is not an integer from -2 to 2"
    if label["event_type"] not in EVENT_TYPES:
        return "event_type is not one of the allowed types"
    confidence = label["confidence"]
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        return "confidence is not a number from 0 to 1"
    return None


def parse_reply(content):
    """Parse and validate the model's reply text. No repair is attempted.

    Returns
    -------
    tuple
        ``(label, None)`` for a valid reply, or ``(None, reason)``.
    """
    try:
        label = json.loads(content)
    except (TypeError, ValueError):
        return None, "reply is not valid JSON"
    reason = validate_label(label)
    return (None, reason) if reason else (label, None)


class ModelUnavailable(Exception):
    """Raised when the API cannot be reached or answers with an error."""


def request_label(session, key, title, summary):
    """Send one article to the pinned model.

    Only the message ``content`` is returned; any other message field (such
    as a separate reasoning field) is ignored.

    Returns
    -------
    tuple
        ``(content, model_reported, usage)``: the reply text; the ``model``
        field of the API response (empty string if the response omits it;
        the key is replaced by ``[redacted]`` should the server ever echo it
        there); and the token counts from the response's ``usage``, as a dict
        with the keys in ``TOKEN_FIELDS`` (0 for a count that is missing or
        not an integer).

    Raises
    ------
    ModelUnavailable
        On connection errors, timeouts, HTTP error statuses (including rate
        limits) and responses without a reply. The message never contains the
        key or the response body.
    """
    payload = {"model": MODEL, "temperature": TEMPERATURE, "messages": build_messages(title, summary)}
    try:
        response = session.post(ENDPOINT, json=payload, timeout=TIMEOUT_S,
                                headers={"Authorization": f"Bearer {key}"})
    except requests.RequestException as exc:
        raise ModelUnavailable(f"request failed: {type(exc).__name__}") from None
    if response.status_code == 429:
        raise ModelUnavailable("HTTP 429 rate limited")
    if not 200 <= response.status_code < 300:
        raise ModelUnavailable(f"HTTP {response.status_code}")
    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise ModelUnavailable("response has no reply message") from None
    return content, str(body.get("model") or "").replace(key, "[redacted]"), token_usage(body)


def token_usage(body):
    """Return the token counts of an API response as a dict with the keys in ``TOKEN_FIELDS``.

    A count that is missing or not an integer is 0.
    """
    usage = body.get("usage") if isinstance(body, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    return {field: usage[field] if _is_int(usage.get(field)) and usage[field] >= 0 else 0
            for field in TOKEN_FIELDS}


def reported_model_error(model_reported):
    """Return an error message if the API reported a model other than ``EXPECTED_REPORTED_MODEL``.

    Returns None when the reported model is the expected one.
    """
    if model_reported == EXPECTED_REPORTED_MODEL:
        return None
    return f"reported model mismatch: expected {EXPECTED_REPORTED_MODEL!r}, got {model_reported!r}"


def _read_csv(path):
    """Return the rows of a CSV file as dicts, or an empty list if it does not exist."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_row(path, columns, row):
    """Append one row to a CSV file, writing the header if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _label_row(article_id, model_reported, labeled_utc, label):
    """Format a validated label as a ``labels.csv`` row."""
    row = {"article_id": article_id, "model_requested": MODEL, "model_reported": model_reported,
           "prompt_version": PROMPT_VERSION, "labeled_utc": labeled_utc}
    for field in LABEL_FIELDS:
        value = label[field]
        if isinstance(value, bool):
            value = "true" if value else "false"
        row[field] = "" if value is None else value
    return row


def label_articles(key, data_dir=DEFAULT_DATA_DIR, session=None, limit=None,
                   now=None, sleep=time.sleep):
    """Label unlabeled articles, oldest publication time first.

    An article is skipped if ``labels.csv`` already holds a label for it with
    the same ``MODEL`` and ``PROMPT_VERSION``. Invalid replies are logged and
    not stored, so the article is retried on a later run. If the model is
    unavailable, or the API reports a model other than
    ``EXPECTED_REPORTED_MODEL``, nothing is stored for that article, the run
    stops and the remaining articles are left for the next run. Requests are
    spaced ``BACKOFF_S`` seconds apart. The run row records the totals of
    ``TOKEN_FIELDS`` over every response received in the run, including
    invalid and rejected replies.

    Parameters
    ----------
    key : str
        The API key. Never logged or written to disk.
    data_dir : Path
        The project data directory; files are in ``data_dir / "news"``.
    session : object or None
        Anything with a ``requests``-style ``post`` method. Inject a fake in
        tests. Defaults to a new ``requests.Session``.
    limit : int or None
        Maximum number of articles to attempt in this run.
    now : callable or None
        ``now() -> datetime`` (timezone-aware). Defaults to the current UTC time.
    sleep : callable
        ``sleep(seconds)`` used for the back-off between requests.

    Returns
    -------
    dict
        The run-log row, as written to ``label_runs.csv``.
    """
    now = now or (lambda: datetime.now(timezone.utc))
    session = session or requests.Session()
    news_dir = Path(data_dir) / "news"
    labels_path = news_dir / "labels.csv"

    done = {row["article_id"] for row in _read_csv(labels_path)
            if row["model_requested"] == MODEL and row["prompt_version"] == PROMPT_VERSION}
    articles = sorted((a for a in _read_csv(news_dir / "articles.csv") if a["id"] not in done),
                      key=lambda a: a["published_utc"])
    if limit is not None:
        articles = articles[:limit]

    run = {"run_utc": format_utc(now()), "attempted": 0, "labeled": 0, "invalid": 0,
           "failed": 0, "error": "", **{field: 0 for field in TOKEN_FIELDS}}
    for position, article in enumerate(articles):
        if position:
            sleep(BACKOFF_S)
        run["attempted"] += 1
        try:
            summary = strip_boilerplate(article["source"], article["summary"])
            content, model_reported, usage = request_label(session, key, article["title"], summary)
        except ModelUnavailable as exc:
            logger.warning("model unavailable, stopping run: %s", exc)
            run["failed"] += 1
            run["error"] = str(exc)
            break
        for field in TOKEN_FIELDS:
            run[field] += usage[field]
        mismatch = reported_model_error(model_reported)
        if mismatch:
            logger.warning("article %s: %s; reply not stored, stopping run", article["id"], mismatch)
            run["failed"] += 1
            run["error"] = mismatch
            break
        label, reason = parse_reply(content)
        if label is None:
            logger.warning("invalid reply for article %s: %s", article["id"], reason)
            run["invalid"] += 1
            continue
        _append_row(labels_path, LABEL_COLUMNS,
                    _label_row(article["id"], model_reported, format_utc(now()), label))
        run["labeled"] += 1

    _append_row(news_dir / "label_runs.csv", RUN_COLUMNS, run)
    return run


def main(argv=None):
    """Command-line entry point: label unlabeled articles and print the counts."""
    parser = argparse.ArgumentParser(description="Label collected news articles with an LLM.")
    parser.add_argument("--limit", type=int, default=None, help="maximum articles to attempt")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="project data directory (default: %(default)s)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run = label_articles(read_env_key(), data_dir=args.data_dir, limit=args.limit)
    print(f"model {MODEL}, prompt_version {PROMPT_VERSION}")
    for column in RUN_COLUMNS[1:]:
        print(f"{column:<10}{run[column]}")


if __name__ == "__main__":
    main()
