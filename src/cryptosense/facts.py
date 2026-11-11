"""Build the assistant's fact sheet from real data.

Every fact is computed here, in tested code, from data the dashboard already
loads. The AI assistant only reads these facts; it never computes or guesses
numbers itself.
"""

import math

import pandas as pd

from cryptosense.context import describe_swing
from cryptosense.one_day import RV_OPEN_THRESHOLDS

# Development-period terciles of rv_open (Phase 13 pre-registration; notebooks 15-16).
# The upper tercile is the pre-registered threshold in ``one_day.RV_OPEN_THRESHOLDS``,
# so "high" activity here means exactly "above the setup threshold" there.
US_OPEN_LOWER_TERCILES = {"BTC/USDT": 0.005763, "SOL/USDT": 0.012285}
US_OPEN_TERCILES = {symbol: (low, RV_OPEN_THRESHOLDS[symbol]) for symbol, low in US_OPEN_LOWER_TERCILES.items()}

# What each forecast method has shown on held-out data (Phase 14, notebook 17).
FORECAST_TRACK_RECORD = {
    "BTC/USDT": "HAR forecast: on 2024+ held-out data it was 21% more accurate than the previous forecast "
    "for total price swings (including moves within the day), but not better for where the price closes.",
    "SOL/USDT": "HAR forecast: on 2024+ held-out data it was 11% more accurate than the previous forecast "
    "for total price swings (including moves within the day), but not better for where the price closes.",
}
FALLBACK_TRACK_RECORD = "20-day volatility forecast (scaled); no intraday data is used for this asset."


def _clean(value):
    """Convert NaN or missing values to None, and NumPy numbers to plain floats."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    return None if math.isnan(value) else value


def _change(close, days):
    """Simple % change of the close over the last `days` days, or None without enough history."""
    if len(close) <= days:
        return None
    return _clean(close.iloc[-1] / close.iloc[-1 - days] - 1)


def classify_us_open(symbol, rv_open):
    """'low', 'middle' or 'high' US-open activity for the asset, or None if unavailable."""
    rv_open = _clean(rv_open)
    if symbol not in US_OPEN_TERCILES or rv_open is None:
        return None
    low, high = US_OPEN_TERCILES[symbol]
    if rv_open > high:
        return "high"
    if rv_open > low:
        return "middle"
    return "low"


def build_facts(daily, symbol, swing_1d, swing_method, now=None, rv_open=None, patterns=None):
    """Facts about the latest complete day, for the assistant and the insights.

    Parameters
    ----------
    daily : pandas.DataFrame
        Output of add_regimes(add_features(load_processed(...))), indexed by UTC date.
    symbol : str
        e.g. "SOL/USDT".
    swing_1d : float
        Tomorrow's typical swing (daily volatility as a fraction), from the dashboard forecast.
    swing_method : str
        Label of the forecast method used (from dashboard_forecast).
    now : pandas.Timestamp, optional
        Current time (UTC); defaults to the real current time. Passed in so tests can fix it.
    rv_open : float, optional
        Realized volatility of the latest day's 09:00-10:59 New York window.
    patterns : list of dict, optional
        Active patterns, each with "name", "description" and "evidence" (badge text).

    Returns
    -------
    dict
        Plain Python values only (no NaN); missing information is None.
    """
    if daily.empty:
        raise ValueError("daily data is empty")
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    last = daily.iloc[-1]
    last_date = daily.index[-1]
    close = daily["close"]
    recent = daily.iloc[-30:]
    price = _clean(last["close"])
    swing = _clean(swing_1d)

    facts = {
        "asset": symbol,
        "data_date": str(last_date.date()),
        "data_age_days": int((now.normalize() - last_date.normalize()).days),
        "price": price,
        "change_1d": _change(close, 1),
        "change_7d": _change(close, 7),
        "change_30d": _change(close, 30),
        "high_30d": _clean(recent["high"].max() if "high" in recent else recent["close"].max()),
        "low_30d": _clean(recent["low"].min() if "low" in recent else recent["close"].min()),
        "trend": last.get("trend") if isinstance(last.get("trend"), str) else None,
        "vol_regime": last.get("vol_regime") if isinstance(last.get("vol_regime"), str) else None,
        "swing_1d": swing,
        "swing_method": swing_method,
        "forecast_track_record": FORECAST_TRACK_RECORD.get(symbol, FALLBACK_TRACK_RECORD)
        if swing_method and swing_method.startswith("HAR")
        else FALLBACK_TRACK_RECORD,
        "us_open_activity": classify_us_open(symbol, rv_open),
        "next_weekday": (last_date + pd.Timedelta(days=1)).day_name(),
        "active_patterns": list(patterns or []),
    }
    if price and swing:
        rng = describe_swing(price, swing)
        facts["range_low"], facts["range_high"] = rng["low"], rng["high"]
    else:
        facts["range_low"] = facts["range_high"] = None
    return facts
