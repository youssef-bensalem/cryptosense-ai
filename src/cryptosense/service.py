"""Everything an interface shows, built from the tested analysis modules.

Each function returns plain Python values (dicts, lists, numbers, strings and
None; never NaN), ready to be sent as JSON. No function here contains
analysis rules of its own: it loads data, calls the tested modules and
arranges their results. The web API (``cryptosense.api``) and the Streamlit
dashboard both read from here, so the two interfaces cannot disagree.

Loading is cached per data file and invalidated when the file changes on
disk, so repeated requests do not re-read or recompute anything.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cryptosense.assistant import ask, is_direction_question, is_trade_question
from cryptosense.backtest import COST_PER_SIDE, run_backtest, summarize_backtest
from cryptosense.context import derive_insights
from cryptosense.data import load_processed, symbol_to_slug
from cryptosense.explain import (
    describe_setup, describe_trend, describe_volatility, evidence_gap, evidence_meaning, position_size,
    setup_evidence, stop_level, suggested_stop, typical_move_range,
)
from cryptosense.facts import build_facts
from cryptosense.features import add_features
from cryptosense.one_day import us_open_measures
from cryptosense.regimes import add_regimes
from cryptosense.setups import SETUPS, explain_setup, setup_flag
from cryptosense.volatility import (
    HAR_LABEL, HAR_SYMBOLS, daily_realized_vol, dashboard_forecast, realized_vol_ahead, rms_ahead,
)

WATCHLIST = ["SOL/USDT", "BTC/USDT", "ETH/USDT"]
ASSET_NAMES = {"SOL/USDT": "Solana", "BTC/USDT": "Bitcoin", "ETH/USDT": "Ethereum"}
CHART_RANGES = {"1m": pd.DateOffset(months=1), "3m": pd.DateOffset(months=3),
                "1y": pd.DateOffset(years=1), "all": None}
DEFAULT_RANGE = "3m"
WEEK_HORIZON = 5
INTRADAY_DAYS = 3
HELD_OUT_START = "2024-01-01"
STRATEGY_PERIODS = {"full": None, "heldout": HELD_OUT_START}
STRATEGY_HOLDS = (5, 10)
DAYS_PER_YEAR = 365
DISCLAIMER = "Research and decision support only. Not financial advice; no trades are placed."

REPLICATION_NOTES = {
    "ETH/USDT": (
        "Setups were designed on BTC/USDT. On ETH/USDT (notebook 08), neither setup showed an "
        "advantage over random trades taken during uptrends."
    ),
    "SOL/USDT": (
        "Setups were designed on BTC/USDT. On SOL/USDT (notebook 08), the continuation setup beat "
        "an average day of the same year in every year since 2021, but the advantage has shrunk "
        "from about +10% to about +2% per 5 days in recent years. The pullback setup underperformed."
    ),
}
PATTERNS_CAPTION = (
    "Setups are research candidates, not buy or sell signals. Rules were pre-registered "
    "and evaluated on BTC/USDT; historical statistics describe the past and are not a forecast."
)
CONDITIONS_CAPTION = (
    "Trend: up = above the 100-day average with positive 20-day momentum; "
    "down = both negative; range = they disagree. "
    "Volatility: high/low relative to its median over the last 365 days."
)
HAR_CAPTION = (
    "HAR forecast from 15-minute realized volatility. On 2024+ held-out data it improved forecast "
    "error on realized volatility by 21% (BTC) and 11% (SOL) versus the previous forecast; it was "
    "not better at forecasting where the price closes."
)
SCALED_CAPTION = (
    "Forecast = 20-day volatility × a scale factor learned only from past, completed outcomes. "
    "In Phase 4 (BTC), this scaled forecast beat raw 20-day volatility by 8–11% in every "
    "validation and test year; a Random Forest performed comparably and was not adopted."
)
NO_INTRADAY_NOTE = "No intraday data for this asset; using the 20-day volatility forecast."
STALE_INTRADAY_NOTE = "15-minute data is missing or out of date; using the 20-day volatility forecast."
STRATEGY_CAPTION = (
    "Continuation setup, long only: enter at the next day's open after a signal, hold 5 or 10 days, "
    f"one position at a time, all-in, {2 * COST_PER_SIDE:.2%} round-trip costs. Past performance does not "
    "guarantee future results."
)
BTC_STRATEGY_FINDINGS = (
    "Phase 6 findings (BTC): the continuation strategy stayed profitable after costs on held-out "
    "data, with far smaller drawdowns and a better return per unit of risk than buy-and-hold, but "
    "a lower total return. Its advantage over random trades taken during uptrends is suggestive, "
    "not established; trade counts are small."
)
MODEL_UNAVAILABLE = (
    "The chat model is not available right now, so I can only answer questions about buying, selling "
    "or price direction (with a fixed, checked answer). Everything else on this page still works."
)
WITHHELD_PREFIX = "*The assistant's answer was withheld"


class UnknownAsset(KeyError):
    """The requested asset is not on the watchlist."""


class NoData(LookupError):
    """The asset has no downloaded data yet."""


# --- small helpers -----------------------------------------------------------

def clean(value):
    """Convert a result into plain JSON-safe Python values.

    NaN, infinities, ``None`` and pandas missing values become None; NumPy
    numbers become plain numbers; timestamps and dates become ISO strings;
    dicts, lists and tuples are converted recursively.
    """
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def slug_to_symbol(slug):
    """Return the watchlist symbol for a slug such as ``"sol_usdt"``.

    Raises
    ------
    UnknownAsset
        If no watchlist symbol has that slug.
    """
    for symbol in WATCHLIST:
        if symbol_to_slug(symbol) == slug:
            return symbol
    raise UnknownAsset(slug)


def _in_range(frame, chart_range):
    """Rows of ``frame`` within the chart range, counted back from its last row."""
    if chart_range not in CHART_RANGES:
        raise ValueError(f"Unknown range {chart_range!r}: expected one of {sorted(CHART_RANGES)}.")
    offset = CHART_RANGES[chart_range]
    return frame if offset is None or frame.empty else frame.loc[frame.index[-1] - offset:]


def _points(frame):
    """A DataFrame as a list of row dicts with the index as ``t`` (ISO date or time)."""
    daily = len(frame) > 0 and bool((frame.index == frame.index.normalize()).all())
    rows = []
    for stamp, row in zip(frame.index, frame.to_dict("records")):
        rows.append({"t": str(stamp.date()) if daily else stamp.isoformat(), **row})
    return clean(rows)


# --- cached loading ----------------------------------------------------------

_CACHE = {}


def clear_cache():
    """Forget everything loaded so far (used after a data refresh and in tests)."""
    _CACHE.clear()


def _path(data_dir, symbol, timeframe):
    return Path(data_dir) / "processed" / f"{symbol_to_slug(symbol)}_{timeframe}.csv"


def _stamp(path):
    """A value that changes whenever the file changes, or None if it does not exist."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _cached(name, data_dir, symbol, extra, build):
    """Return ``build()``, reusing the last result while the asset's data files are unchanged."""
    stamps = (_stamp(_path(data_dir, symbol, "1d")), _stamp(_path(data_dir, symbol, "15m")))
    key = (name, str(data_dir), symbol, extra)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == stamps:
        return hit[1]
    result = build()
    _CACHE[key] = (stamps, result)
    return result


def has_data(data_dir, symbol):
    """True if the asset's daily data has been downloaded."""
    return _path(data_dir, symbol, "1d").exists()


def load_daily(data_dir, symbol):
    """Daily data with features and regimes.

    Raises
    ------
    NoData
        If the asset's daily file does not exist.
    """
    if not has_data(data_dir, symbol):
        raise NoData(symbol)
    return _cached("daily", data_dir, symbol, None,
                   lambda: add_regimes(add_features(load_processed(symbol, Path(data_dir)))))


def load_m15(data_dir, symbol):
    """15-minute candles for assets that use them, or None if not downloaded."""
    if symbol not in HAR_SYMBOLS or not _path(data_dir, symbol, "15m").exists():
        return None
    return _cached("m15", data_dir, symbol, None,
                   lambda: load_processed(symbol, Path(data_dir), timeframe="15m"))


def load_forecast(data_dir, symbol, horizon):
    """The dashboard's volatility forecast over ``horizon`` days and its method label."""
    return _cached("forecast", data_dir, symbol, horizon,
                   lambda: dashboard_forecast(symbol, load_daily(data_dir, symbol), load_m15(data_dir, symbol),
                                              horizon=horizon))


def active_patterns(data_dir, symbol):
    """Setups flagged on the latest day: name, plain-language description and evidence label."""
    def build():
        df = load_daily(data_dir, symbol)
        patterns = []
        for setup in SETUPS:
            explanation = explain_setup(df, df.index[-1], setup)
            if not explanation["flagged"]:
                continue
            evidence = setup_evidence(explanation)
            if symbol != "BTC/USDT":
                evidence += " (designed on BTC)"
            patterns.append({"name": setup, "description": describe_setup(setup), "evidence": evidence})
        return patterns
    return _cached("active_patterns", data_dir, symbol, None, build)


def _rv_open(data_dir, symbol, date):
    """US-open realized volatility on ``date``, or None (no 15-minute data, or none that day)."""
    m15 = load_m15(data_dir, symbol)
    if m15 is None:
        return None
    measures = _cached("us_open", data_dir, symbol, None, lambda: us_open_measures(m15))
    value = measures["rv_open"].get(date)
    return None if value is None or pd.isna(value) else float(value)


def facts_and_insights(data_dir, symbol, now=None):
    """The fact sheet for tomorrow and the deductions drawn from it (see ``facts`` and ``context``)."""
    df = load_daily(data_dir, symbol)
    forecast, method = load_forecast(data_dir, symbol, 1)
    facts = build_facts(df, symbol, forecast.iloc[-1], method, now=now,
                        rv_open=_rv_open(data_dir, symbol, df.index[-1]),
                        patterns=active_patterns(data_dir, symbol))
    return facts, derive_insights(facts)


# --- payloads ----------------------------------------------------------------

def list_assets(data_dir, now=None):
    """One entry per watchlist asset: identity, whether data exists, and the latest price facts."""
    assets = []
    for symbol in WATCHLIST:
        entry = {"symbol": symbol, "slug": symbol_to_slug(symbol), "name": ASSET_NAMES[symbol],
                 "has_data": has_data(data_dir, symbol)}
        if entry["has_data"]:
            facts, _ = facts_and_insights(data_dir, symbol, now)
            entry.update({key: facts[key] for key in
                          ("data_date", "data_age_days", "price", "change_1d", "change_7d", "change_30d")})
        assets.append(entry)
    return clean(assets)


def today(data_dir, symbol, now=None):
    """The "Today & tomorrow" view: tomorrow's typical swing, deductions, and recent 15-minute prices."""
    df = load_daily(data_dir, symbol)
    facts, insights = facts_and_insights(data_dir, symbol, now)
    last_date = df.index[-1]
    method = facts["swing_method"]
    is_har = method == HAR_LABEL
    m15 = load_m15(data_dir, symbol)
    if m15 is None:
        intraday = None
    else:
        recent = m15.loc[m15.index[-1] - pd.Timedelta(days=INTRADAY_DAYS):]
        intraday = _points(recent[["close"]])
    ordered = sorted(insights, key=lambda insight: insight.kind != "warning")
    return clean({
        "symbol": symbol,
        "name": ASSET_NAMES[symbol],
        "data_date": facts["data_date"],
        "data_age_days": facts["data_age_days"],
        "tomorrow": str((last_date + pd.Timedelta(days=1)).date()),
        "next_weekday": facts["next_weekday"],
        "price": facts["price"],
        "change_1d": facts["change_1d"],
        "change_7d": facts["change_7d"],
        "change_30d": facts["change_30d"],
        "low_30d": facts["low_30d"],
        "high_30d": facts["high_30d"],
        "trend": facts["trend"],
        "trend_words": describe_trend(facts["trend"]),
        "vol_regime": facts["vol_regime"],
        "vol_words": describe_volatility(facts["vol_regime"]),
        "method": method,
        "swing_1d": facts["swing_1d"] if facts["range_low"] is not None else None,
        "swing_word": "swing" if is_har else "move",
        "swing_name": "typical swing (including moves within the day)" if is_har else "typical move",
        "range_low": facts["range_low"],
        "range_high": facts["range_high"],
        "forecast_track_record": facts["forecast_track_record"],
        "insights": [{"title": i.title, "text": i.text, "kind": i.kind, "evidence": i.evidence} for i in ordered],
        "intraday": intraday,
        "intraday_supported": symbol in HAR_SYMBOLS,
        "disclaimer": "This describes conditions and risk. It is not advice to buy or sell.",
    })


def week(data_dir, symbol):
    """The "This week" view: trend and turbulence in words, the typical 5-day move, and patterns."""
    df = load_daily(data_dir, symbol)
    forecast, method = load_forecast(data_dir, symbol, WEEK_HORIZON)
    last_date = df.index[-1]
    latest = df.iloc[-1]
    close = latest["close"]
    is_har = method == HAR_LABEL
    daily_vol = forecast.iloc[-1]
    if pd.isna(daily_vol):
        move = low = high = None
    else:
        move, low, high = typical_move_range(close, daily_vol, horizon=WEEK_HORIZON)
    distance = latest["dist_sma_100"]
    patterns = []
    for setup in SETUPS:
        explanation = explain_setup(df, last_date, setup)
        entry = {"name": setup, "description": describe_setup(setup), "present": explanation["flagged"],
                 "evidence": None, "meaning": None}
        if explanation["flagged"]:
            entry["evidence"] = setup_evidence(explanation)
            entry["meaning"] = evidence_meaning(entry["evidence"])
        patterns.append(entry)
    return clean({
        "symbol": symbol,
        "data_date": str(last_date.date()),
        "price": close,
        "trend": latest["trend"] if isinstance(latest["trend"], str) else None,
        "trend_words": describe_trend(latest["trend"]),
        "vol_regime": latest["vol_regime"] if isinstance(latest["vol_regime"], str) else None,
        "vol_words": describe_volatility(latest["vol_regime"]),
        "change_20d": np.exp(latest["mom_20"]) - 1,
        "above_sma_100": None if pd.isna(distance) else bool(distance >= 0),
        "horizon_days": WEEK_HORIZON,
        "method": method,
        "swing_word": "swing" if is_har else "move",
        "swing_name": "typical swing (including moves within the day)" if is_har else "typical move",
        "move": move,
        "range_low": low,
        "range_high": high,
        "patterns": patterns,
        "replication_note": REPLICATION_NOTES.get(symbol),
        "disclaimer": "This summary describes market conditions and past evidence. It is not advice to buy or sell.",
    })


def prices(data_dir, symbol, chart_range=DEFAULT_RANGE):
    """Daily close, volume and trend label over the chart range, with the headline price changes."""
    df = load_daily(data_dir, symbol)
    close = df["close"]

    def change(days):
        return close.iloc[-1] / close.iloc[-1 - days] - 1 if len(close) > days else None

    return clean({
        "symbol": symbol,
        "range": chart_range,
        "last_close": close.iloc[-1],
        "change_1d": change(1),
        "change_30d": change(30),
        "points": _points(_in_range(df, chart_range)[["close", "volume", "trend"]]),
    })


def conditions(data_dir, symbol):
    """The "Market conditions" view: current regimes, the numbers behind them, and regime shares."""
    df = load_daily(data_dir, symbol)
    latest = df.iloc[-1]
    shares = df["trend"].value_counts(normalize=True)
    return clean({
        "symbol": symbol,
        "data_date": str(df.index[-1].date()),
        "trend": latest["trend"] if isinstance(latest["trend"], str) else None,
        "vol_regime": latest["vol_regime"] if isinstance(latest["vol_regime"], str) else None,
        "momentum_20d": np.exp(latest["mom_20"]) - 1,
        "distance_sma_100": latest["dist_sma_100"],
        "volatility_20d_annual": latest["vol_20"] * np.sqrt(DAYS_PER_YEAR),
        "efficiency_ratio_20d": latest["er_20"],
        "trend_shares": {label: shares.get(label, 0.0) for label in ("up", "range", "down")},
        "caption": CONDITIONS_CAPTION,
    })


def patterns(data_dir, symbol):
    """The "Patterns (research)" view: each setup's rules today and its historical record."""
    df = load_daily(data_dir, symbol)
    forecast, method = load_forecast(data_dir, symbol, WEEK_HORIZON)
    last_date = df.index[-1]
    typical = forecast.iloc[-1] * np.sqrt(WEEK_HORIZON)
    setups = []
    for setup in SETUPS:
        e = explain_setup(df, last_date, setup)
        gap, gap_se = evidence_gap(e)
        flags = setup_flag(df, setup)
        starts = flags & ~flags.shift(fill_value=False)
        has_record = e["past_episodes"] > 1
        setups.append({
            "name": setup,
            "description": describe_setup(setup),
            "flagged": e["flagged"],
            "date": e["date"],
            "trend": e["trend"] if isinstance(e["trend"], str) else None,
            "conditions": e["conditions"],
            "past_episodes": e["past_episodes"],
            "past_mean_return": e["past_mean_return"] if has_record else None,
            "past_mean_se": e["past_mean_se"] if has_record else None,
            "base_same_trend_mean": e["base_same_trend_mean"] if has_record else None,
            "gap": gap if has_record else None,
            "gap_in_standard_errors": gap / gap_se if has_record and gap_se else None,
            "recent_starts": [str(day.date()) for day in starts[starts].index[-5:]],
        })
    return clean({
        "symbol": symbol,
        "typical_move_5d": typical,
        "swing_word": "swing" if method == HAR_LABEL else "move",
        "method": method,
        "setups": setups,
        "caption": PATTERNS_CAPTION,
        "replication_note": REPLICATION_NOTES.get(symbol),
    })


def volatility(data_dir, symbol, chart_range=DEFAULT_RANGE):
    """The "Volatility" view: the 5-day forecast, how it is made, and forecast against realized."""
    df = load_daily(data_dir, symbol)
    forecast, method = load_forecast(data_dir, symbol, WEEK_HORIZON)
    is_har = method == HAR_LABEL
    latest = forecast.iloc[-1]
    m15 = load_m15(data_dir, symbol)
    if m15 is not None:
        realized = _cached("realized_ahead", data_dir, symbol, WEEK_HORIZON,
                           lambda: realized_vol_ahead(daily_realized_vol(m15), horizon=WEEK_HORIZON)).reindex(df.index)
        realized_name = f"realized next {WEEK_HORIZON} days (15-minute data)"
    else:
        realized = rms_ahead(df["log_return"], WEEK_HORIZON)
        realized_name = f"realized next {WEEK_HORIZON} days"
    series = _in_range(pd.DataFrame({"forecast": forecast, "realized": realized}), chart_range) * np.sqrt(DAYS_PER_YEAR)
    if is_har:
        caption = HAR_CAPTION
    else:
        caption = SCALED_CAPTION + " " + (NO_INTRADAY_NOTE if symbol not in HAR_SYMBOLS else STALE_INTRADAY_NOTE)
    return clean({
        "symbol": symbol,
        "data_date": str(df.index[-1].date()),
        "horizon_days": WEEK_HORIZON,
        "method": method,
        "daily": latest,
        "typical_move": latest * np.sqrt(WEEK_HORIZON),
        "annualised": latest * np.sqrt(DAYS_PER_YEAR),
        "swing_word": "swing" if is_har else "move",
        "caption": caption,
        "realized_name": realized_name,
        "range": chart_range,
        "points": _points(series),
        "note": f"Realized volatility is only known {WEEK_HORIZON} days later, so the most recent days "
                "show the forecast alone.",
    })


def run_strategies(data_dir, symbol, start=None):
    """Backtest the continuation setup (5- and 10-day holds) and buy-and-hold from ``start``.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        The summary table (one row per strategy) and the equity curves (one column per strategy).
        Both are empty when fewer than three days have every feature the setup needs.
    """
    def build():
        data = load_daily(data_dir, symbol).loc[start:].dropna(subset=["dist_sma_100", "rel_volume_20", "er_20"])
        if len(data) < 3:
            return pd.DataFrame(), pd.DataFrame()
        summaries, curves = {}, {}
        for hold in STRATEGY_HOLDS:
            trades, equity = run_backtest(data, setup_flag(data, "continuation"), hold, COST_PER_SIDE)
            summaries[f"continuation {hold}d"] = summarize_backtest(trades, equity)
            curves[f"continuation {hold}d"] = equity
        hold_all = pd.Series(False, index=data.index)
        hold_all.iloc[0] = True
        trades, equity = run_backtest(data, hold_all, len(data) - 2, COST_PER_SIDE)
        summary = summarize_backtest(trades, equity)
        summary["win_rate"] = summary["avg_trade"] = None
        summaries["buy and hold"] = summary
        curves["buy and hold"] = equity
        return pd.DataFrame(summaries).T, pd.DataFrame(curves)
    return _cached("strategies", data_dir, symbol, start, build)


def strategies(data_dir, symbol, period="full", chart_range=DEFAULT_RANGE):
    """The "Strategy test results" view: the summary table and equity curves for a period."""
    if period not in STRATEGY_PERIODS:
        raise ValueError(f"Unknown period {period!r}: expected one of {sorted(STRATEGY_PERIODS)}.")
    table, curves = run_strategies(data_dir, symbol, STRATEGY_PERIODS[period])
    rows = [{"name": name, **row} for name, row in table.to_dict("index").items()]
    return clean({
        "symbol": symbol,
        "period": period,
        "range": chart_range,
        "strategies": list(curves.columns),
        "summary": rows,
        "points": _points(_in_range(curves, chart_range)),
        "caption": STRATEGY_CAPTION,
        "findings": BTC_STRATEGY_FINDINGS if symbol == "BTC/USDT" else None,
        "replication_note": REPLICATION_NOTES.get(symbol),
    })


def risk(price, typical_move, account, risk_pct):
    """Suggested stop and position size for one trade (see ``explain.position_size``).

    Parameters
    ----------
    price : float
        Current price.
    typical_move : float
        Typical move over the holding period, as a log return.
    account : float
        Account size.
    risk_pct : float
        Share of the account to risk on the trade, in percent (1 means 1%).

    Raises
    ------
    ValueError
        If any input is not a positive, finite number, or ``risk_pct`` is above 100.
    """
    for name, value in [("price", price), ("typical_move", typical_move), ("account", account),
                        ("risk_pct", risk_pct)]:
        if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be a positive number; got {value!r}.")
    if risk_pct > 100:
        raise ValueError(f"risk_pct must be at most 100; got {risk_pct!r}.")
    stop_price, stop_fraction = stop_level(price, suggested_stop(typical_move))
    invest = position_size(account, risk_pct / 100, stop_fraction)
    return clean({
        "stop_price": stop_price,
        "stop_fraction": stop_fraction,
        "invest": invest,
        "loss_at_stop": account * risk_pct / 100,
        "exceeds_account": invest > account,
    })


def answer(data_dir, symbol, question, history=None, client=None, now=None):
    """Answer a question about an asset, and say where the answer came from.

    Returns
    -------
    dict
        ``answer`` (Markdown text) and ``source``: ``"fixed"`` for the checked
        answer to buy/sell and direction questions, ``"model"`` for a language
        model's answer that passed the guard, ``"withheld"`` when the guard
        replaced the model's answer, ``"unavailable"`` when no model is
        configured or it failed.
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    facts, insights = facts_and_insights(data_dir, symbol, now)
    if is_trade_question(question) or is_direction_question(question):
        return {"answer": ask(question, facts, insights, client), "source": "fixed"}
    if client is None:
        return {"answer": MODEL_UNAVAILABLE, "source": "unavailable"}
    text = ask(question, facts, insights, client, history=history)
    if text.startswith(WITHHELD_PREFIX):
        source = "withheld"
    elif text.startswith("The assistant is unavailable right now"):
        source = "unavailable"
    else:
        source = "model"
    return {"answer": text, "source": source}
