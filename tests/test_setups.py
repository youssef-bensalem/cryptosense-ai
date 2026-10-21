import numpy as np
import pandas as pd
import pytest

from cryptosense.features import add_features
from cryptosense.regimes import add_regimes
from cryptosense.setups import explain_setup, setup_flag
from test_features import make_random_df


def make_tiny_df():
    """30 daily rows with pullback episodes starting on days 5 and 27."""
    index = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC", name="date")
    close = pd.Series(100 * np.exp(np.arange(30) * 0.01), index=index)
    dist_sma_20 = pd.Series(0.1, index=index)
    dist_sma_20.iloc[[5, 27]] = -0.1
    return pd.DataFrame({
        "close": close,
        "log_return": np.log(close / close.shift(1)),
        "vol_20": 0.02,
        "dist_sma_100": 0.1,
        "dist_sma_20": dist_sma_20,
        "trend": "up",
        "vol_regime": "low",
    }, index=index)


def test_flags_match_rules():
    df = add_regimes(add_features(make_random_df(800)))

    pullback = (df["dist_sma_100"] > 0) & (df["dist_sma_20"] < 0)
    continuation = (df["mom_20"] > 0) & (df["er_20"] > 0.4) & (df["rel_volume_20"] > 1.5)

    assert setup_flag(df, "pullback").equals(pullback)
    assert setup_flag(df, "continuation").equals(continuation)


def test_conditions_known_values():
    df = make_tiny_df()

    flagged = explain_setup(df, df.index[27], "pullback")
    not_flagged = explain_setup(df, df.index[28], "pullback")

    assert flagged["flagged"] is True
    assert [c["met"] for c in flagged["conditions"]] == [True, True]
    assert flagged["conditions"][1]["value"] == -0.1
    assert not_flagged["flagged"] is False


def test_only_completed_outcomes_are_counted():
    df = make_tiny_df()

    result = explain_setup(df, df.index[29], "pullback", horizon=5)

    # Episode on day 5 is complete by day 29; episode on day 27 is not.
    assert result["past_episodes"] == 1


@pytest.mark.parametrize("position", [450, 600, 750])
def test_explanation_uses_only_past_data(position):
    df = add_regimes(add_features(make_random_df(800)))
    date = df.index[position]

    full = explain_setup(df, date, "continuation")
    truncated = explain_setup(df.loc[:date], date, "continuation")

    assert full == truncated