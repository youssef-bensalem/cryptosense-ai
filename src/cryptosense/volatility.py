"""Point-in-time volatility forecasts: scaled 20-day volatility (Phase 4's most consistent
forecast), daily realized volatility from 15-minute data, and HAR (Phase 14)."""

import numpy as np
import pandas as pd


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


HAR_COEFFICIENTS = ["intercept", "b_log_rv_1", "b_log_rv_5", "b_log_rv_22"]


def _har_design(daily_rv, horizon):
    """HAR features and fitting target on a complete daily calendar.

    Dates absent from ``daily_rv``'s index are treated as missing RV, so every
    window covers consecutive calendar days.

    Returns
    -------
    tuple of (pandas.DatetimeIndex, numpy.ndarray, numpy.ndarray)
        The calendar days; features (n x 3: log RV_t, log mean RV over t-4..t,
        log mean RV over t-21..t); and the target log sqrt(mean RV^2 over
        t+1..t+horizon). Missing windows give NaN (or -inf for zero RV).
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not (daily_rv.index.is_monotonic_increasing and daily_rv.index.is_unique):
        raise ValueError("daily_rv must have a sorted, unique date index")
    days = pd.date_range(daily_rv.index[0], daily_rv.index[-1], freq="D").as_unit(daily_rv.index.unit)
    rv = daily_rv.astype(float).reindex(days)
    with np.errstate(divide="ignore"):
        features = np.column_stack([np.log(rv), np.log(rv.rolling(5).mean()), np.log(rv.rolling(22).mean())])
        target = np.log(np.sqrt((rv ** 2).rolling(horizon).mean()).shift(-horizon)).to_numpy()
    return days, features, target


def _fit_ols(features, target):
    """Least-squares intercept and slopes of target on features."""
    design = np.column_stack([np.ones(len(features)), features])
    return np.linalg.lstsq(design, target, rcond=None)[0]


def fit_har_latest(daily_rv, horizon=5):
    """Fit HAR on every row whose features and ``horizon``-day target are fully known.

    The regression is log of the next-``horizon``-day realized volatility,
    sqrt(mean of RV^2 over days t+1..t+horizon), on log RV_t, log of the mean
    RV over days t-4..t and log of the mean RV over days t-21..t. Rows with any
    missing value in these windows are excluded.

    Parameters
    ----------
    daily_rv : pandas.Series
        Daily realized volatility indexed by UTC date (output of ``daily_realized_vol``).
    horizon : int, default 5
        Target horizon in days.

    Returns
    -------
    pandas.Series
        ``intercept``, ``b_log_rv_1``, ``b_log_rv_5`` and ``b_log_rv_22``.
    """
    _, features, target = _har_design(daily_rv, horizon)
    usable = np.isfinite(features).all(axis=1) & np.isfinite(target)
    if usable.sum() < len(HAR_COEFFICIENTS):
        raise ValueError("not enough usable rows to fit HAR")
    coef = _fit_ols(features[usable], target[usable])
    return pd.Series(coef, index=HAR_COEFFICIENTS, name="har_coefficients")


def har_forecast(daily_rv, horizon=5, refit_every=30, min_train=365):
    """Point-in-time HAR forecast of daily volatility over the next ``horizon`` days.

    Features and fitting target are as in ``fit_har_latest``. The model used on
    day s is fitted only on rows i with i + horizon <= s, so every target it
    learns from is complete by the close of day s. The first model is fitted on
    the first day with at least ``min_train`` such usable rows; after that the
    model is refitted every ``refit_every`` calendar days and reused in between.

    Parameters
    ----------
    daily_rv : pandas.Series
        Daily realized volatility indexed by UTC date (output of ``daily_realized_vol``).
        Dates absent from the index count as missing RV.
    horizon : int, default 5
        Forecast horizon in days.
    refit_every : int, default 30
        Days between refits.
    min_train : int, default 365
        Usable rows needed before the first fit (at least 4).

    Returns
    -------
    pandas.Series
        exp of the predicted log volatility, aligned with ``daily_rv``'s index,
        named ``har_forecast``; NaN where today's features are missing or no
        model exists yet.
    """
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1")
    if min_train < len(HAR_COEFFICIENTS):
        raise ValueError(f"min_train must be at least {len(HAR_COEFFICIENTS)}")
    days, features, target = _har_design(daily_rv, horizon)
    has_features = np.isfinite(features).all(axis=1)
    usable = has_features & np.isfinite(target)
    # known_count[s]: usable rows i with i + horizon <= s.
    known_count = np.concatenate([np.zeros(horizon, dtype=int), np.cumsum(usable)])[:len(days)]

    forecast = np.full(len(days), np.nan)
    coef, last_fit = None, None
    for s in range(len(days)):
        due = coef is None or s - last_fit >= refit_every
        if due and known_count[s] >= min_train:
            rows = usable.copy()
            rows[max(s - horizon + 1, 0):] = False
            coef, last_fit = _fit_ols(features[rows], target[rows]), s
        if coef is not None and has_features[s]:
            forecast[s] = np.exp(coef[0] + features[s] @ coef[1:])
    return pd.Series(forecast, index=days, name="har_forecast").reindex(daily_rv.index)


HAR_SYMBOLS = ("BTC/USDT", "SOL/USDT")
HAR_LABEL = "HAR (realized volatility)"
SCALED_LABEL = "20-day volatility (scaled)"


def realized_vol_ahead(daily_rv, horizon=5):
    """Realized volatility over the next ``horizon`` days, sqrt(mean of RV^2 over days t+1..t+horizon).

    This is the outcome HAR forecasts, for comparing forecasts with what
    happened. It uses future data by design and must never be used as a
    feature. Dates absent from the index count as missing RV.

    Parameters
    ----------
    daily_rv : pandas.Series
        Daily realized volatility indexed by UTC date (output of ``daily_realized_vol``).
    horizon : int, default 5
        Number of days ahead.

    Returns
    -------
    pandas.Series
        Aligned with ``daily_rv``'s index, named ``realized_vol_ahead``; NaN
        where any of the next ``horizon`` days is missing or not yet known.
    """
    _, _, target = _har_design(daily_rv, horizon)
    days = pd.date_range(daily_rv.index[0], daily_rv.index[-1], freq="D").as_unit(daily_rv.index.unit)
    return pd.Series(np.exp(target), index=days, name="realized_vol_ahead").reindex(daily_rv.index)


def dashboard_forecast(symbol, daily, m15=None, horizon=5):
    """The dashboard's daily volatility forecast for one asset, and the method used.

    BTC/USDT and SOL/USDT with 15-minute data use ``har_forecast`` on
    ``daily_realized_vol(m15)``; other assets, or no 15-minute data, use
    ``scaled_vol_forecast``. If HAR has no value on the latest daily date
    (for example, stale 15-minute data), that date falls back to the scaled
    forecast and the label says so.

    Parameters
    ----------
    symbol : str
        Market symbol in CCXT format, e.g. ``"BTC/USDT"``.
    daily : pandas.DataFrame
        Daily data with ``log_return`` and ``vol_20`` columns, indexed by UTC date.
    m15 : pandas.DataFrame, optional
        15-minute candles from ``load_processed(..., timeframe="15m")``.
    horizon : int, default 5
        Forecast horizon in days.

    Returns
    -------
    tuple of (pandas.Series, str)
        Forecast daily volatility aligned with ``daily``'s index, named
        ``vol_forecast``, and the method label.
    """
    scaled = scaled_vol_forecast(daily, horizon)
    if symbol not in HAR_SYMBOLS or m15 is None:
        return scaled, SCALED_LABEL

    rv = daily_realized_vol(m15)
    har = har_forecast(rv, horizon=horizon)
    har = har.set_axis(har.index.as_unit(daily.index.unit)).reindex(daily.index).rename("vol_forecast")
    if pd.isna(har.iloc[-1]):
        har.iloc[-1] = scaled.iloc[-1]
        return har, f"{SCALED_LABEL} on {daily.index[-1].date()}: HAR unavailable on that date"
    return har, HAR_LABEL
