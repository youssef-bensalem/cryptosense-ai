import numpy as np
import pandas as pd
import pytest

from cryptosense.data import DAY_MS, candles_to_dataframe
from cryptosense.features import add_features

START_MS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC

FEATURE_WARMUP = {
    "log_return": 1,
    "mom_5": 5,
    "mom_20": 20,
    "dist_sma_20": 19,
    "dist_sma_100": 99,
    "vol_20": 20,
    "vol_60": 60,
    "rel_volume_20": 20,
    "er_20": 20,
}


def make_random_df(n=300, seed=0):
    """Return a DataFrame of n daily candles following a random walk."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    volume = rng.lognormal(10, 0.4, n)
    candles = [
        [START_MS + i * DAY_MS, c, c * 1.02, c * 0.98, c, v]
        for i, (c, v) in enumerate(zip(close, volume))
    ]
    return candles_to_dataframe(candles)


def test_adds_exactly_the_feature_columns():
    df = make_random_df()
    out = add_features(df)

    new_columns = [c for c in out.columns if c not in df.columns]
    assert new_columns == list(FEATURE_WARMUP)


@pytest.mark.parametrize("feature, warmup", FEATURE_WARMUP.items())
def test_warmup_nan_count(feature, warmup):
    out = add_features(make_random_df())

    assert out[feature].isna().sum() == warmup
    assert out[feature].iloc[:warmup].isna().all()


@pytest.mark.parametrize("cutoff", [30, 100, 150, 250])
def test_features_use_only_past_data(cutoff):
    df = make_random_df()

    full = add_features(df)
    truncated = add_features(df.iloc[:cutoff])

    pd.testing.assert_frame_equal(full.iloc[:cutoff], truncated)


def test_does_not_modify_input():
    df = make_random_df()
    before = df.copy()
    add_features(df)

    pd.testing.assert_frame_equal(df, before)