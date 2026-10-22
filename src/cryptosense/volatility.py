"""Point-in-time volatility forecast: scaled 20-day volatility (Phase 4's most consistent forecast)."""

import numpy as np


def scaled_vol_forecast(df, horizon=5):
    """Forecast daily volatility over the next ``horizon`` days for every day.

    The forecast is ``vol_20`` multiplied by a scale factor equal to the
    average ratio of realized ``horizon``-day volatility to ``vol_20`` over
    past days whose outcome was already complete. On day t, only ratios from
    days up to t - horizon are used, so no future information enters.

    Parameters
    ----------
    df : pandas.DataFrame
        Daily data with ``log_return`` and ``vol_20`` columns.
    horizon : int, default 5
        Forecast horizon in days.

    Returns
    -------
    pandas.Series
        Forecast daily volatility; NaN until enough completed outcomes exist.
    """
    realized = np.sqrt((df["log_return"] ** 2).rolling(horizon).mean()).shift(-horizon)
    log_ratio = np.log(realized / df["vol_20"])
    scale = np.exp(log_ratio.expanding().mean().shift(horizon))
    return (df["vol_20"] * scale).rename("vol_forecast")

CANDLES_PER_DAY = 96  # 15-minute candles in a UTC day


def daily_realized_vol(m15, max_missing=4):
    """Daily realized volatility from 15-minute candles.

    For each UTC day, the square root of the sum of squared 15-minute log
    returns within that day. Each candle's return is log(close / open), so
    every return covers exactly its own 15 minutes inside the day: no
    return crosses a day boundary or depends on a neighbouring candle. A
    complete day has 96 candles; a day with more than ``max_missing``
    missing candles gets NaN, because too much of its price path is
    unobserved.

    Parameters
    ----------
    m15 : pandas.DataFrame
        15-minute candles from ``load_processed(..., timeframe="15m")``,
        indexed by UTC open time, with ``open`` and ``close`` columns.
    max_missing : int, default 4
        Largest number of missing candles a day may have and still get a value.

    Returns
    -------
    pandas.Series
        Daily realized volatility (log-return units, comparable with the
        standard deviation of daily log returns), indexed by UTC day for
        every day with at least one candle, named ``rv_daily``.
    """
    r = np.log(m15["close"] / m15["open"])
    day = m15.index.normalize()
    present = r.notna().groupby(day).sum()
    rv = np.sqrt((r ** 2).groupby(day).sum())
    rv = rv.where(CANDLES_PER_DAY - present <= max_missing)
    rv.index.name = "date"
    return rv.rename("rv_daily")
