"""Pre-registered one-day setups D1 and D2 (Phase 13).

A signal at the daily close of day t buys at the open of day t + 1 and sells
at the open of day t + 2 (see ``find_one_day_trades``). The rules were
pre-registered and approved in notebooks/15_one_day_setups.ipynb before any
outcome was computed; they are defined once here.
"""

import operator

import numpy as np
import pandas as pd

from cryptosense.intraday import CANDLE, NY_TZ, lag_by_time

US_OPEN_HOURS_NY = (9, 10)
CANDLES_IN_US_OPEN = 8
EXCLUDED_ENTRY_WEEKDAY = "Saturday"

# Development-period (to 2023-12-31) 2/3 quantiles of rv_open, printed in notebooks 15 and 16.
RV_OPEN_THRESHOLDS = {"BTC/USDT": 0.010688, "SOL/USDT": 0.021055}
RV_OPEN_THRESHOLD = "rv_open threshold of the asset"

ONE_DAY_SETUPS = {
    "D1": [("trend", "==", "up"), ("rv_open", ">", RV_OPEN_THRESHOLD), ("ret_open", ">", 0.0)],
    "D2": [("rv_open", ">", RV_OPEN_THRESHOLD), ("ret_open", "<", 0.0), ("log_return", ">", 0.0)],
}
ONE_DAY_OPS = {">": operator.gt, "<": operator.lt, "==": operator.eq}


def us_open_measures(m15):
    """Activity and price change around the US open on each UTC day.

    The window is the 8 15-minute candles opening from 09:00 to 10:45 New
    York time (13:00-14:45 UTC in US summer time, 14:00-15:45 UTC
    otherwise), so it ends at 11:00 New York, before the daily close at
    00:00 UTC. Each candle's return is its close against the previous
    candle's close, as in ``add_intraday_features``; the first return
    therefore also needs the 08:45 candle.

    rv_open
        Square root of the sum of the 8 squared returns.
    ret_open
        Sum of the 8 returns: log(close at 11:00 / close at 09:00 New York).

    A day whose window lacks any of the 8 returns (a missing candle,
    including the 08:45 candle) gets NaN for both.

    Parameters
    ----------
    m15 : pandas.DataFrame
        15-minute candles from ``load_processed(..., timeframe="15m")``,
        indexed by UTC open time, with a ``close`` column.

    Returns
    -------
    pandas.DataFrame
        Indexed by UTC day (midnight), with ``rv_open`` and ``ret_open``,
        for every day that has at least one window candle.
    """
    log_close = np.log(m15["close"])
    ret_15m = log_close - lag_by_time(log_close, CANDLE)
    in_window = np.isin(m15.index.tz_convert(NY_TZ).hour, US_OPEN_HOURS_NY)

    r = ret_15m[in_window]
    day = m15.index[in_window].normalize()
    complete = r.groupby(day).count() == CANDLES_IN_US_OPEN
    out = pd.DataFrame({
        "rv_open": np.sqrt((r ** 2).groupby(day).sum()),
        "ret_open": r.groupby(day).sum(),
    }).where(complete)
    out.index.name = "date"
    return out


def one_day_signals(daily, open_measures, symbol):
    """Daily signal of each one-day setup for one asset.

    Day t signals when every condition of the setup in ``ONE_DAY_SETUPS``
    holds on day t, using the asset's own ``rv_open`` threshold from
    ``RV_OPEN_THRESHOLDS``, and the entry day t + 1 (calendar day) is not
    a Saturday. A day with any missing input never signals. Inputs for
    day t are known at its close: the daily features and labels of day t
    and the US-open measures of day t, whose window ends at 11:00 New York.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily data indexed by UTC date with ``trend`` and ``log_return``,
        e.g. ``add_regimes(add_features(load_processed(...)))``.
    open_measures : pandas.DataFrame
        Output of ``us_open_measures`` for the same asset.
    symbol : str
        Asset in CCXT format; must be a key of ``RV_OPEN_THRESHOLDS``.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``daily``, one bool column per setup (True on signal days).

    Raises
    ------
    ValueError
        If ``symbol`` has no pre-registered threshold.
    """
    if symbol not in RV_OPEN_THRESHOLDS:
        raise ValueError(f"No pre-registered rv_open threshold for {symbol!r}; "
                         f"expected one of {sorted(RV_OPEN_THRESHOLDS)}.")
    threshold = RV_OPEN_THRESHOLDS[symbol]

    measures = open_measures.set_axis(open_measures.index.as_unit(daily.index.unit))
    frame = daily[["trend", "log_return"]].join(measures[["rv_open", "ret_open"]])
    entry_weekday = (daily.index + pd.Timedelta(days=1)).day_name()
    not_saturday = pd.Series(entry_weekday != EXCLUDED_ENTRY_WEEKDAY, index=daily.index)

    out = pd.DataFrame(index=daily.index)
    for name, rules in ONE_DAY_SETUPS.items():
        fires = frame[[feature for feature, _, _ in rules]].notna().all(axis=1)
        for feature, op, value in rules:
            value = threshold if value == RV_OPEN_THRESHOLD else value
            fires &= ONE_DAY_OPS[op](frame[feature], value).fillna(False).astype(bool)
        out[name] = fires & not_saturday
    return out
