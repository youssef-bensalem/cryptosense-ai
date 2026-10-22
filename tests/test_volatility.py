import numpy as np
import pandas as pd
import pytest

from cryptosense.features import add_features
from cryptosense.regimes import add_regimes
from cryptosense.setups import explain_setup
from cryptosense.data import candles_to_dataframe
from cryptosense.volatility import daily_realized_vol, fit_har_latest, har_forecast, scaled_vol_forecast
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


TRUE_HAR = {"intercept": -0.5, "b_log_rv_1": 0.3, "b_log_rv_5": 0.3, "b_log_rv_22": 0.25}


def make_har_rv(n, seed=0, start="2020-01-01", noise=0.2):
    """Daily RV from a HAR process in logs: log RV_(t+1) = c + b1 log RV_t + b5 log mean RV_(t-4..t)
    + b22 log mean RV_(t-21..t) + noise, so with horizon 1 the true coefficients are TRUE_HAR."""
    rng = np.random.default_rng(seed)
    c, b1, b5, b22 = TRUE_HAR.values()
    rv = list(np.exp(rng.normal(-3.3, 0.2, 22)))
    while len(rv) < n:
        log_next = (c + b1 * np.log(rv[-1]) + b5 * np.log(np.mean(rv[-5:]))
                    + b22 * np.log(np.mean(rv[-22:])) + rng.normal(0, noise))
        rv.append(np.exp(log_next))
    return pd.Series(rv, index=pd.date_range(start, periods=n, freq="D", tz="UTC"), name="rv_daily")


def test_fit_har_latest_recovers_known_coefficients():
    rv = make_har_rv(6000)
    coef = fit_har_latest(rv, horizon=1)
    assert list(coef.index) == list(TRUE_HAR)
    for name, value in TRUE_HAR.items():
        assert coef[name] == pytest.approx(value, abs=0.06), name


@pytest.mark.parametrize("cut", [150, 333, 480, 799])
def test_har_forecast_uses_only_past_data(cut):
    rv = make_har_rv(800)
    full = har_forecast(rv, refit_every=30, min_train=100)
    truncated = har_forecast(rv.iloc[:cut + 1], refit_every=30, min_train=100)

    assert truncated.notna().sum() > 0
    pd.testing.assert_series_equal(full.iloc[:cut + 1], truncated)


def test_har_forecast_ignores_values_after_each_day():
    rv = make_har_rv(400)
    base = har_forecast(rv, refit_every=30, min_train=100)
    first = base.first_valid_index()
    s0 = rv.index.get_loc(first)
    # Every day of two refit cycles, so refit days are included.
    for s in range(s0, s0 + 60):
        extreme = rv.copy()
        extreme.iloc[s + 1:] = 50.0
        changed = har_forecast(extreme, refit_every=30, min_train=100)
        pd.testing.assert_series_equal(changed.iloc[:s + 1], base.iloc[:s + 1])

    # Control: the same change applied from day s itself does change day s.
    extreme = rv.copy()
    extreme.iloc[s0:] = 50.0
    assert har_forecast(extreme, refit_every=30, min_train=100).iloc[s0] != base.iloc[s0]


def test_har_forecast_matches_latest_fit_between_refits():
    rv = make_har_rv(400)
    forecast = har_forecast(rv, refit_every=30, min_train=100)
    s0 = rv.index.get_loc(forecast.first_valid_index())
    for refit in [s0, s0 + 30, s0 + 60]:
        coef = fit_har_latest(rv.iloc[:refit + 1])
        for s in range(refit, refit + 30):
            x = [np.log(rv.iloc[s]), np.log(rv.iloc[s - 4:s + 1].mean()), np.log(rv.iloc[s - 21:s + 1].mean())]
            expected = np.exp(coef["intercept"] + np.dot(coef.iloc[1:], x))
            assert forecast.iloc[s] == pytest.approx(expected, rel=1e-10)


def test_har_missing_rv_excluded_from_features_and_fit():
    rv = make_har_rv(300)
    k, horizon = 150, 5
    gappy = rv.copy()
    gappy.iloc[k] = np.nan

    # Features at rows k..k+21 include day k, so today's forecast there is NaN.
    forecast = har_forecast(gappy, horizon=horizon, refit_every=30, min_train=100)
    assert forecast.iloc[k:k + 22].isna().all()
    assert forecast.iloc[k + 22:].notna().all()
    assert forecast.iloc[k - horizon:k].notna().all()

    # The fit uses exactly the rows whose features and target avoid day k.
    rows = [i for i in range(21, len(rv) - horizon)
            if not (i - 21 <= k <= i) and not (i + 1 <= k <= i + horizon)]
    x = np.array([[1.0, np.log(rv.iloc[i]), np.log(rv.iloc[i - 4:i + 1].mean()),
                   np.log(rv.iloc[i - 21:i + 1].mean())] for i in rows])
    y = np.array([np.log(np.sqrt((rv.iloc[i + 1:i + 1 + horizon] ** 2).mean())) for i in rows])
    expected = np.linalg.lstsq(x, y, rcond=None)[0]
    np.testing.assert_allclose(fit_har_latest(gappy, horizon=horizon).to_numpy(), expected, rtol=1e-10)

    # A date missing from the index counts as missing RV.
    dropped = rv.drop(index=rv.index[k])
    pd.testing.assert_series_equal(har_forecast(dropped, refit_every=30, min_train=100),
                                   forecast.drop(index=rv.index[k]))
    pd.testing.assert_series_equal(fit_har_latest(dropped), fit_har_latest(gappy))

    # Missing RV today: no forecast today.
    last_missing = rv.copy()
    last_missing.iloc[-1] = np.nan
    assert np.isnan(har_forecast(last_missing, refit_every=30, min_train=100).iloc[-1])


def test_har_forecast_nan_before_min_train():
    rv = make_har_rv(300)
    min_train, horizon = 100, 5
    forecast = har_forecast(rv, horizon=horizon, refit_every=30, min_train=min_train)
    # Usable rows start at 21 (22-day window); row i is known on day i + horizon.
    first = 21 + min_train - 1 + horizon
    assert forecast.iloc[:first].isna().all()
    assert forecast.iloc[first:].notna().all()


def test_har_does_not_modify_input():
    rv = make_har_rv(300)
    rv.iloc[100] = np.nan
    snapshot = rv.copy(deep=True)
    har_forecast(rv, refit_every=30, min_train=100)
    fit_har_latest(rv)
    pd.testing.assert_series_equal(rv, snapshot)
