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