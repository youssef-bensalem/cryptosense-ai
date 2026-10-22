import numpy as np
import pandas as pd
import pytest

from cryptosense.features import add_features
from cryptosense.regimes import add_regimes
from cryptosense.setups import explain_setup
from cryptosense.data import candles_to_dataframe
from cryptosense.volatility import daily_realized_vol, scaled_vol_forecast
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


def make_m15_days(n_days, rng=None, start="2024-01-01"):
    """n_days of complete 15-minute candles with random per-candle log returns (close / open)."""
    rng = np.random.default_rng(0) if rng is None else rng
    n = n_days * 96
    r = rng.normal(0, 0.003, n)
    close = 100 * np.exp(np.cumsum(r))
    open_ = close / np.exp(r)
    start_ms = int(pd.Timestamp(start, tz="UTC").value // 10**6)
    candles = [[start_ms + i * 900_000, o, max(o, c), min(o, c), c, 1.0]
               for i, (o, c) in enumerate(zip(open_, close))]
    return candles_to_dataframe(candles), r


def test_daily_realized_vol_known_values():
    m15, r = make_m15_days(3)
    rv = daily_realized_vol(m15)
    expected = np.sqrt((r.reshape(3, 96) ** 2).sum(axis=1))
    assert list(rv.index) == list(pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"))
    np.testing.assert_allclose(rv.to_numpy(), expected, rtol=1e-12)

    # Constant per-candle return of 0.001: sqrt(96) x 0.001.
    start_ms = int(pd.Timestamp("2024-01-01", tz="UTC").value // 10**6)
    flat = candles_to_dataframe([[start_ms + i * 900_000, 100.0, 100.2, 99.9, 100.0 * np.exp(0.001), 1.0]
                                 for i in range(96)])
    assert daily_realized_vol(flat).iloc[0] == pytest.approx(np.sqrt(96) * 0.001)


@pytest.mark.parametrize("n_missing, has_value", [(4, True), (5, False)])
def test_daily_realized_vol_missing_candles(n_missing, has_value):
    m15, r = make_m15_days(2)
    drop = np.arange(96 + 10, 96 + 10 + n_missing)          # candles missing on day 2 only
    rv = daily_realized_vol(m15.drop(index=m15.index[drop]))

    assert rv.iloc[0] == pytest.approx(np.sqrt(np.sum(r[:96] ** 2)))
    if has_value:
        kept = np.delete(r[96:], drop - 96)
        assert rv.iloc[1] == pytest.approx(np.sqrt(np.sum(kept ** 2)))
    else:
        assert np.isnan(rv.iloc[1])


def test_daily_realized_vol_never_crosses_a_day_boundary():
    m15, r = make_m15_days(3)
    before = daily_realized_vol(m15)
    # A large jump between the last candle of day 1 and the first candle of day 2,
    # outside both days' candles, changes neither day.
    jumped = m15.copy()
    jumped.iloc[96:, [jumped.columns.get_loc(c) for c in ["open", "high", "low", "close"]]] *= 1.5
    pd.testing.assert_series_equal(daily_realized_vol(jumped), before)
    # Each day depends only on its own candles.
    for d in range(3):
        alone = daily_realized_vol(m15.iloc[d * 96:(d + 1) * 96])
        assert alone.iloc[0] == pytest.approx(before.iloc[d])


def test_daily_realized_vol_does_not_modify_input():
    m15, _ = make_m15_days(2)
    snapshot = m15.copy(deep=True)
    daily_realized_vol(m15)
    pd.testing.assert_frame_equal(m15, snapshot)
