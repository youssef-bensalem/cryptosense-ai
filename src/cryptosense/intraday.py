"""Intraday features for 15-minute candles (Phase 9).

Timing rule: the row for a candle that opens at time t is known at
t + 15 minutes (its close), and every feature on that row uses only
candles that open at or before t. Windows and lags are defined by time,
not by row count: a feature whose required past candle is missing is NaN,
never computed from a different time.
"""

import numpy as np
import pandas as pd

CANDLE = pd.Timedelta("15min")
RETURN_WINDOWS = {"ret_1h": "1h", "ret_4h": "4h", "ret_24h": "24h"}
RV_WINDOWS = {"rv_4h": "4h", "rv_24h": "24h"}
REL_VOLUME_DAYS = 20
NY_TZ = "America/New_York"


def lag_by_time(series, lag):
    """Value of ``series`` exactly ``lag`` earlier, aligned to its own index.

    NaN where no row exists at that earlier time.

    Parameters
    ----------
    series : pandas.Series
        Indexed by a DatetimeIndex.
    lag : str or pandas.Timedelta
        How far back to look, e.g. ``"1h"``.

    Returns
    -------
    pandas.Series
        Same index as ``series``.
    """
    return series.shift(freq=pd.Timedelta(lag)).reindex(series.index)


def window_sum(series, window):
    """Sum of ``series`` over the time window (t - window, t], or NaN if incomplete.

    The window is complete only when every 15-minute candle in it exists and
    has a non-missing value, i.e. it holds ``window / 15min`` values.

    Parameters
    ----------
    series : pandas.Series
        Values on 15-minute candles, indexed by a DatetimeIndex.
    window : str or pandas.Timedelta
        Window length, a multiple of 15 minutes.

    Returns
    -------
    pandas.Series
        Same index as ``series``.
    """
    window = pd.Timedelta(window)
    rolling = series.rolling(window)
    expected = window // CANDLE
    return rolling.sum().where(rolling.count() == expected)


def add_intraday_features(m15, daily):
    """Add intraday features to 15-minute candles, using only past information.

    Each row is the candle that opens at time t; all features on it use only
    candles that open at or before t, so they are known at t + 15 minutes.

    Columns added
    -------------
    ret_1h, ret_4h, ret_24h
        Log of the close divided by the close of the candle that opened
        exactly 1h / 4h / 24h earlier.
    rv_4h, rv_24h
        Square root of the sum of squared 15-minute log returns (close to
        previous close) over the candles opening in (t - 4h, t] / (t - 24h, t].
    rel_volume_1h
        Volume over the candles opening in (t - 1h, t], divided by the mean
        of that same quantity at the same UTC clock time on each of the
        previous 20 days. NaN if that mean is 0.
    hour_utc
        Hour of t in UTC.
    hour_ny
        Hour of t in New York local time (America/New_York, with daylight
        saving time).
    prev_day_trend, prev_day_vol_regime
        The daily ``trend`` and ``vol_regime`` labels of the previous UTC
        day. A candle on day D gets day D-1's labels, because day D's labels
        use day D's close.

    A feature is NaN when any candle it needs is missing; no value is taken
    from a different time.

    Parameters
    ----------
    m15 : pandas.DataFrame
        15-minute candles from ``load_processed(..., timeframe="15m")``,
        indexed by UTC open time, with ``close`` and ``volume`` columns.
    daily : pandas.DataFrame
        Daily data indexed by UTC date with ``trend`` and ``vol_regime``
        columns, e.g. ``add_regimes(add_features(load_processed(...)))``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``m15`` with the columns above appended. The inputs are
        not modified.
    """
    out = m15.copy()
    close = out["close"]
    log_close = np.log(close)

    for column, window in RETURN_WINDOWS.items():
        out[column] = log_close - lag_by_time(log_close, window)

    ret_15m = log_close - lag_by_time(log_close, CANDLE)
    for column, window in RV_WINDOWS.items():
        out[column] = np.sqrt(window_sum(ret_15m ** 2, window))

    volume_1h = window_sum(out["volume"], "1h")
    past = pd.concat(
        [lag_by_time(volume_1h, pd.Timedelta(days=k)) for k in range(1, REL_VOLUME_DAYS + 1)],
        axis=1,
    )
    past_mean = past.mean(axis=1, skipna=False)
    out["rel_volume_1h"] = volume_1h / past_mean.where(past_mean > 0)

    out["hour_utc"] = out.index.hour
    out["hour_ny"] = out.index.tz_convert(NY_TZ).hour

    prev_day = out.index.normalize() - pd.Timedelta(days=1)
    for column, label in [("prev_day_trend", "trend"), ("prev_day_vol_regime", "vol_regime")]:
        labels = daily[label]
        labels = labels.set_axis(labels.index.as_unit(prev_day.unit))
        out[column] = labels.reindex(prev_day).to_numpy()

    return out
