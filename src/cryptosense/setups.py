"""Rule-based trading setups: flags and point-in-time explanations.

Setups are research candidates, not buy or sell signals. All rules are
defined once in ``SETUPS`` and were pre-registered before any outcomes
were examined.
"""

import operator

import numpy as np
import pandas as pd

from cryptosense.volatility import scaled_vol_forecast

SETUPS = {
    "pullback": [("dist_sma_100", ">", 0.0), ("dist_sma_20", "<", 0.0)],
    "continuation": [("mom_20", ">", 0.0), ("er_20", ">", 0.4), ("rel_volume_20", ">", 1.5)],
}
# Comparison operators for every rule table in the project (daily, intraday and one-day setups).
OPS = {">": operator.gt, "<": operator.lt, "==": operator.eq}


def conditions_hold(frame, rules):
    """True on the rows of ``frame`` where every rule holds.

    Parameters
    ----------
    frame : pandas.DataFrame
        Data containing the features named by the rules.
    rules : list of tuple
        ``(feature, operator, value)`` with an operator from ``OPS``.

    Returns
    -------
    pandas.Series of bool
        Aligned with ``frame``'s index. A row with a missing value in any
        feature used is False.
    """
    holds = pd.Series(True, index=frame.index)
    for feature, op, value in rules:
        holds &= OPS[op](frame[feature], value).fillna(False).astype(bool)
    return holds


def setup_flag(frame, setup):
    """Flag the days on which every condition of a setup holds.

    Rules come from ``SETUPS``: each condition is a (feature, operator,
    threshold) tuple, and a day is flagged only if all of them are met.
    Days with a missing feature value are never flagged.

    Parameters
    ----------
    frame : pandas.DataFrame
        Data containing the features used by the setup's rules.
    setup : str
        A key of ``SETUPS``, e.g. ``"continuation"``.

    Returns
    -------
    pandas.Series of bool
        True on flagged days, aligned with ``frame``'s index.
    """

    return conditions_hold(frame, SETUPS[setup])


def explain_setup(df, date, setup, horizon=5):
    """Explain a setup on one day, using only data up to and including that day.

    Point-in-time guarantee: everything is computed from ``df.loc[:date]``,
    so no information from after ``date`` is used. Historical results only
    include past episodes whose ``horizon``-day outcome was complete by
    ``date``.

    Requires the output of ``add_regimes(add_features(...))``: the features
    used by the setup's rules, plus ``close``, ``log_return``, ``vol_20``,
    ``trend`` and ``vol_regime``.

    Parameters
    ----------
    df : pandas.DataFrame
        Daily data with features and regimes, indexed by UTC date.
    date : str or pandas.Timestamp
        The day to explain.
    setup : str
        A key of ``SETUPS``.
    horizon : int, default 5
        Outcome horizon in days, used for the volatility forecast and the
        historical results.

    Returns
    -------
    dict
        The conditions with their values and whether each was met; the
        current trend and volatility regimes; a volatility forecast
        (``scaled_vol_forecast``) and the typical ``horizon``-day move; and statistics of
        past episode outcomes, with their standard error and two base rates
        (all past days, and past days in the same trend regime).

    Notes
    -----
    The historical statistics describe the past; they are not a forecast.
    Judge a setup by comparing ``past_mean_return`` with
    ``base_same_trend_mean`` relative to ``past_mean_se``. The comparison
    with all past days overstates any advantage, because setups tend to
    fire in strong markets.
    """

    history = df.loc[:date]
    today = history.iloc[-1]

    conditions = [
        {"feature": f, "rule": f"{op} {t}", "value": round(float(today[f]), 4), "met": bool(OPS[op](today[f], t))}
        for f, op, t in SETUPS[setup]
    ]

    vol_forecast = scaled_vol_forecast(history, horizon).iloc[-1]

    flag = setup_flag(history, setup)
    starts = flag & ~flag.shift(fill_value=False)
    fwd = np.log(history["close"].shift(-horizon) / history["close"])
    past = fwd[starts].dropna()
    base = fwd.dropna()
    same_trend = fwd[history["trend"] == today["trend"]].dropna()

    return {
        "date": history.index[-1].date(),
        "setup": setup,
        "flagged": all(c["met"] for c in conditions),
        "conditions": conditions,
        "trend": today["trend"],
        "vol_regime": today["vol_regime"],
        "daily_vol_forecast": round(float(vol_forecast), 4),
        "typical_move": round(float(vol_forecast * np.sqrt(horizon)), 4),
        "past_episodes": int(len(past)),
        "past_mean_return": round(float(past.mean()), 4) if len(past) else None,
        "past_share_positive": round(float((past > 0).mean()), 3) if len(past) else None,
        "past_mean_se": round(float(past.std() / np.sqrt(len(past))), 4) if len(past) > 1 else None,
        "base_mean_return": round(float(base.mean()), 4),
        "base_same_trend_mean": round(float(same_trend.mean()), 4) if len(same_trend) else None,
    }
