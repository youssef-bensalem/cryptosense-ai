import numpy as np
import pandas as pd
import pytest

from cryptosense.features import add_features
from cryptosense.regimes import add_regimes
from test_features import make_random_df


def test_trend_rule_known_values():
    df = pd.DataFrame({
        "dist_sma_100": [0.10, -0.10, 0.10, -0.10, np.nan],
        "mom_20":       [0.05, -0.05, -0.05, 0.05, 0.05],
        "vol_20":       [0.02] * 5,
    })
    out = add_regimes(df)

    assert out["trend"].iloc[:4].tolist() == ["up", "down", "range", "range"]
    assert pd.isna(out["trend"].iloc[4])


def test_volatility_rule_known_values():
    vol = [0.02] * 400 + [0.05, 0.01]
    df = pd.DataFrame({"dist_sma_100": 0.0, "mom_20": 0.0, "vol_20": vol})
    out = add_regimes(df)

    assert out["vol_regime"].iloc[-2] == "high"
    assert out["vol_regime"].iloc[-1] == "low"


def test_warmup_nan_counts():
    out = add_regimes(add_features(make_random_df(800)))

    assert out["trend"].isna().sum() == 99
    assert out["vol_regime"].isna().sum() == 384


def test_does_not_modify_input():
    df = add_features(make_random_df(500))
    before = df.copy()
    add_regimes(df)

    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("cutoff", [400, 500, 650])
def test_regimes_use_only_past_data(cutoff):
    df = add_features(make_random_df(800))

    full = add_regimes(df)
    truncated = add_regimes(df.iloc[:cutoff])

    pd.testing.assert_frame_equal(full.iloc[:cutoff], truncated)