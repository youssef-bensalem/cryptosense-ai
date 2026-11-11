"""Plain-language descriptions of market conditions, evidence and risk.

Pure functions with no Streamlit and no data loading, so the dashboard
can show results in everyday words while every rule stays tested here.
"""

import math

import pandas as pd

NOT_ENOUGH_HISTORY = "not enough history yet"
TREND_WORDS = {
    "up": "rising",
    "down": "falling",
    "range": "mixed (the signals disagree)",
}
VOLATILITY_WORDS = {
    "high": "more turbulent than usual (compared with the past year)",
    "low": "calmer than usual (compared with the past year)",
}
SETUP_WORDS = {
    "continuation": "a strong, steady rise with unusually heavy trading",
    "pullback": "a short-term dip while the longer-term trend is rising",
}
MIN_EPISODES = 30
EVIDENCE_LABELS = (
    "Too little data",
    "Did worse than average",
    "No evidence",
    "Weak evidence",
    "Some evidence",
    "Stronger evidence",
)
EVIDENCE_MEANINGS = {
    "Too little data": "This pattern has appeared fewer than 30 times before, too few to judge.",
    "Did worse than average": (
        "In the 5 days after this pattern, prices did worse than on an average day "
        "in the same trend, by more than chance usually explains."
    ),
    "No evidence": (
        "In the 5 days after this pattern, prices did about as well as on an average day "
        "in the same trend; any difference could easily be chance."
    ),
    "Weak evidence": (
        "In the 5 days after this pattern, prices did somewhat better than on an average day "
        "in the same trend, but a difference this size often happens by chance."
    ),
    "Some evidence": (
        "In the 5 days after this pattern, prices did clearly better than on an average day "
        "in the same trend, though chance cannot be ruled out."
    ),
    "Stronger evidence": (
        "In the 5 days after this pattern, prices did much better than on an average day "
        "in the same trend; the past still does not guarantee the future."
    ),
}


def _describe(label, words):
    """Look up a regime label in ``words``; a missing label means not enough history."""
    if label is None or pd.isna(label):
        return NOT_ENOUGH_HISTORY
    if label not in words:
        raise ValueError(f"Unknown label {label!r}: expected one of {sorted(words)} or a missing value.")
    return words[label]


def describe_trend(label):
    """Describe the daily trend label in plain words.

    "up" means prices are rising, "down" means they are falling, and
    "range" means the trend signals disagree, so the picture is mixed.
    A missing label means there is not yet enough price history to tell.

    Parameters
    ----------
    label : str or None
        The ``trend`` label from ``add_regimes``: "up", "down", "range",
        or missing (None or NaN).

    Returns
    -------
    str
        The plain-language description.

    Raises
    ------
    ValueError
        If ``label`` is not one of the known labels and not missing.
    """
    return _describe(label, TREND_WORDS)


def describe_volatility(label):
    """Describe the daily volatility label in plain words.

    "high" means the market has been more turbulent than usual and "low"
    means calmer than usual, both compared with the past year rather
    than with a fixed level. A missing label means there is not yet
    enough price history to compare.

    Parameters
    ----------
    label : str or None
        The ``vol_regime`` label from ``add_regimes``: "high", "low", or
        missing (None or NaN).

    Returns
    -------
    str
        The plain-language description.

    Raises
    ------
    ValueError
        If ``label`` is not one of the known labels and not missing.
    """
    return _describe(label, VOLATILITY_WORDS)


def typical_move_range(price, daily_vol, horizon=5):
    """Typical price move over the next few days, and the price range it implies.

    Daily volatility describes a typical one-day move. Over several days
    moves partly cancel out, so the typical move grows with the square
    root of the number of days: move = daily_vol x sqrt(horizon). The
    range runs from price x exp(-move) to price x exp(+move), which is
    symmetric for log returns. It is a typical range, not a limit: prices
    often move further.

    Parameters
    ----------
    price : float
        Current price.
    daily_vol : float
        Daily volatility as a log-return standard deviation, e.g. 0.03.
    horizon : int, default 5
        Number of days ahead.

    Returns
    -------
    tuple of float
        ``(move, low, high)``: the typical move as a log return, and the
        low and high ends of the price range.
    """
    move = daily_vol * math.sqrt(horizon)
    return move, price * math.exp(-move), price * math.exp(move)


def evidence_strength(gap, se, n_episodes):
    """Label how strong the historical evidence for a setup is.

    The gap is how much better (positive) or worse (negative) past
    episodes did than an average comparable day; the standard error says
    how uncertain that gap is. Their ratio counts how many standard
    errors the gap is from zero. The rules, checked in this order:

    - fewer than 30 past episodes: "Too little data", whatever the gap;
    - the gap is negative and at least 1 standard error below zero:
      "Did worse than average";
    - the gap is within 1 standard error of zero: "No evidence";
    - 1 to under 2 standard errors above zero: "Weak evidence";
    - 2 to under 3 standard errors above zero: "Some evidence";
    - 3 or more standard errors above zero: "Stronger evidence".

    Only these six labels can be returned. Past results describe the past
    and are not a forecast.

    Parameters
    ----------
    gap : float
        Mean outcome of past episodes minus the base rate.
    se : float
        Standard error of the episodes' mean outcome. Must be positive.
    n_episodes : int
        Number of past episodes.

    Returns
    -------
    str
        One of ``EVIDENCE_LABELS``.

    Raises
    ------
    ValueError
        If ``n_episodes`` is at least 30 and ``gap`` is not a finite number
        or ``se`` is not a positive finite number, since no label would
        then be justified.
    """
    if n_episodes < MIN_EPISODES:
        return "Too little data"
    if not (math.isfinite(gap) and math.isfinite(se) and se > 0):
        raise ValueError(f"gap must be finite and se positive and finite; got gap={gap!r}, se={se!r}.")

    z = gap / se
    if gap < 0 and z <= -1:
        return "Did worse than average"
    if abs(z) < 1:
        return "No evidence"
    if z < 2:
        return "Weak evidence"
    if z < 3:
        return "Some evidence"
    return "Stronger evidence"


def position_size(account, risk_fraction, stop_fraction):
    """Amount to invest so that hitting the stop loses a chosen share of the account.

    If the price falls to the stop, the position loses stop_fraction of
    its value. Investing account x risk_fraction / stop_fraction makes
    that loss exactly account x risk_fraction. For example, with a 10,000
    account, 1% risk and a 5% stop, invest 2,000: a 5% fall on 2,000
    loses 100, which is 1% of the account. Costs and slippage are not
    included, and a price gap past the stop can lose more.

    Parameters
    ----------
    account : float
        Account value.
    risk_fraction : float
        Share of the account you accept to lose if the stop is hit, e.g. 0.01.
    stop_fraction : float
        Distance from entry to the stop as a share of the entry price, e.g. 0.05.

    Returns
    -------
    float
        Amount to invest.

    Raises
    ------
    ValueError
        If any input is not a positive number.
    """
    for name, value in [("account", account), ("risk_fraction", risk_fraction), ("stop_fraction", stop_fraction)]:
        if not value > 0:
            raise ValueError(f"{name} must be positive; got {value!r}.")
    return account * risk_fraction / stop_fraction


def suggested_stop(typical_move):
    """Suggested stop distance: twice the typical move.

    A stop closer than the typical move would be hit by ordinary price
    noise, so the suggestion leaves room for two typical moves.

    Parameters
    ----------
    typical_move : float
        Typical move over the holding period, e.g. from ``typical_move_range``.

    Returns
    -------
    float
        Stop distance, in the same units as ``typical_move``.
    """
    return 2 * typical_move


def describe_setup(name):
    """Describe a setup from ``SETUPS`` in plain words.

    "continuation" is a strong, steady rise with unusually heavy trading;
    "pullback" is a short-term dip while the longer-term trend is rising.

    Parameters
    ----------
    name : str
        A key of ``cryptosense.setups.SETUPS``.

    Returns
    -------
    str
        The plain-language description.

    Raises
    ------
    ValueError
        If ``name`` is not a known setup.
    """
    if name not in SETUP_WORDS:
        raise ValueError(f"Unknown setup {name!r}: expected one of {sorted(SETUP_WORDS)}.")
    return SETUP_WORDS[name]


def evidence_meaning(label):
    """One plain sentence explaining an ``evidence_strength`` label.

    Each sentence says how prices did in the 5 days after past
    occurrences of the pattern, compared with an average day in the same
    trend, and how far chance could explain the difference.

    Parameters
    ----------
    label : str
        One of ``EVIDENCE_LABELS``.

    Returns
    -------
    str
        The explanation.

    Raises
    ------
    ValueError
        If ``label`` is not one of ``EVIDENCE_LABELS``.
    """
    if label not in EVIDENCE_MEANINGS:
        raise ValueError(f"Unknown evidence label {label!r}: expected one of {list(EVIDENCE_LABELS)}.")
    return EVIDENCE_MEANINGS[label]


def stop_level(price, stop_move):
    """Stop price and stop distance for a stop placed ``stop_move`` below the price.

    Moves in this module are log returns, like the typical move, so a
    stop ``stop_move`` below the price sits at price x exp(-stop_move).
    As a share of the price, the distance to it is 1 - exp(-stop_move);
    that share is what ``position_size`` needs as ``stop_fraction``.

    Parameters
    ----------
    price : float
        Current price.
    stop_move : float
        Stop distance as a log return, e.g. from ``suggested_stop``.

    Returns
    -------
    tuple of float
        ``(stop_price, stop_fraction)``.
    """
    stop_price = price * math.exp(-stop_move)
    return stop_price, 1 - stop_price / price


def evidence_gap(explanation):
    """Gap between a setup's past episodes and its base rate, with its standard error.

    Parameters
    ----------
    explanation : dict
        Output of ``cryptosense.setups.explain_setup``.

    Returns
    -------
    tuple of float
        ``(gap, se)``: ``past_mean_return`` minus ``base_same_trend_mean``,
        and ``past_mean_se``. Both are NaN when the explanation lacks any of
        the three (too few past episodes, or no past day in the same trend).
    """
    values = [explanation.get(key) for key in ("past_mean_return", "base_same_trend_mean", "past_mean_se")]
    if any(value is None for value in values):
        return float("nan"), float("nan")
    return values[0] - values[1], values[2]


def setup_evidence(explanation):
    """The ``evidence_strength`` label for a setup explanation.

    Parameters
    ----------
    explanation : dict
        Output of ``cryptosense.setups.explain_setup``.

    Returns
    -------
    str
        One of ``EVIDENCE_LABELS``.
    """
    gap, se = evidence_gap(explanation)
    return evidence_strength(gap, se, explanation["past_episodes"])
