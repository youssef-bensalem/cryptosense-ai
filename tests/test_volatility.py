import numpy as np
import pandas as pd
import pytest

from cryptosense.features import add_features
from cryptosense.regimes import add_regimes
from cryptosense.setups import explain_setup
from cryptosense.volatility import scaled_vol_forecast
from test_features import make_random_df


@pytest.mark.parametrize("cutoff", [200, 400, 600])
def test_forecast_uses_only_past_data(cutoff):
    df = add_features(make_random_df(800))

    full = scaled_vol_forecast(df)
    truncated = scaled_vol_forecast(df.iloc[:cutoff])

    pd.testing.assert_series_equal(full.iloc[:cutoff], truncated)


def test_forecast_matches_setup_explanations():
    df = add_regimes(add_features(make_random_df(800)))
    forecast = scaled_vol_forecast(df)

    for position in [450, 600, 799]:
        date = df.index[position]
        explained = explain_setup(df, date, "continuation")["daily_vol_forecast"]
        assert round(float(forecast.loc[date]), 4) == explained