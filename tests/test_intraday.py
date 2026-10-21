import numpy as np
import pandas as pd
import pytest

from cryptosense.data import DAY_MS, candles_to_dataframe
from cryptosense.intraday import add_intraday_features

START_MS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC
CANDLE_MS = 15 * 60 * 1000
FEATURES = ["ret_1h", "ret_4h", "ret_24h", "rv_4h", "rv_24h", "rel_volume_1h",
            "hour_utc", "hour_ny", "prev_day_trend", "prev_day_vol_regime"]


def make_m15(close, volume=None, start_ms=START_MS):
    """Return 15-minute candles with the given closes, starting at start_ms."""
    volume = np.ones(len(close)) if volume is None else volume
    candles = [
        [start_ms + i * CANDLE_MS, c, c * 1.001, c * 0.999, c, v]
        for i, (c, v) in enumerate(zip(close, volume))
    ]
    return candles_to_dataframe(candles)


def make_random_m15(n_days=25, seed=0):
    """Return n_days of 15-minute candles following a random walk."""
    rng = np.random.default_rng(seed)
    n = n_days * 96
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    volume = rng.lognormal(5, 0.5, n)
    return make_m15(close, volume)


def make_daily(n_days=30, start_ms=START_MS):
    """Return daily labels that differ on consecutive days."""
    index = pd.to_datetime([start_ms + i * DAY_MS for i in range(n_days)], unit="ms", utc=True)
    return pd.DataFrame({
        "trend": [f"trend_{i}" for i in range(n_days)],
        "vol_regime": [f"vol_{i}" for i in range(n_days)],
    }, index=index)


def test_adds_exactly_the_feature_columns():
    m15 = make_random_m15(n_days=3)
    out = add_intraday_features(m15, make_daily())

    assert [c for c in out.columns if c not in m15.columns] == FEATURES
    pd.testing.assert_frame_equal(out[m15.columns], m15)


def test_returns_and_rv_known_values_constant_step():
    # Every 15-minute log return is exactly 0.001.
    step = 0.001
    m15 = make_m15(100 * np.exp(step * np.arange(200)))
    out = add_intraday_features(m15, make_daily())

    expected = {"ret_1h": 4 * step, "ret_4h": 16 * step, "ret_24h": 96 * step,
                "rv_4h": np.sqrt(16) * step, "rv_24h": np.sqrt(96) * step}
    first_valid = {"ret_1h": 4, "ret_4h": 16, "ret_24h": 96, "rv_4h": 16, "rv_24h": 96}
    for column, value in expected.items():
        k = first_valid[column]
        assert out[column].iloc[:k].isna().all(), column
        assert np.allclose(out[column].iloc[k:], value), column


def test_returns_and_rv_match_positional_computation_without_gaps():
    m15 = make_random_m15(n_days=3)
    out = add_intraday_features(m15, make_daily())

    log_close = np.log(m15["close"].to_numpy())
    r = np.diff(log_close, prepend=np.nan)
    for column, k in [("ret_1h", 4), ("ret_4h", 16), ("ret_24h", 96)]:
        expected = np.full(len(m15), np.nan)
        expected[k:] = log_close[k:] - log_close[:-k]
        np.testing.assert_allclose(out[column].to_numpy(), expected, rtol=1e-12, err_msg=column)

    for column, k in [("rv_4h", 16), ("rv_24h", 96)]:
        expected = np.full(len(m15), np.nan)
        for i in range(k, len(m15)):
            expected[i] = np.sqrt(np.sum(r[i - k + 1:i + 1] ** 2))
        np.testing.assert_allclose(out[column].to_numpy(), expected, rtol=1e-9, err_msg=column)


def test_rel_volume_known_value():
    n = 22 * 96
    volume = np.ones(n)
    t = 21 * 96 + 40              # a candle on day 21 at 10:00 UTC
    volume[t - 3:t + 1] = 2.0     # the 1h ending with candle t has twice the usual volume
    m15 = make_m15(np.full(n, 100.0), volume)
    out = add_intraday_features(m15, make_daily())

    rel = out["rel_volume_1h"]
    # The first value needs a complete 1h window 20 days earlier.
    first = 20 * 96 + 3
    assert rel.iloc[:first].isna().all()
    assert rel.iloc[first] == pytest.approx(1.0)
    assert rel.iloc[t] == pytest.approx(2.0)
    assert rel.iloc[t + 4] == pytest.approx(1.0)


def test_missing_candle_gives_nan_not_a_value_from_another_time():
    m15 = make_random_m15(n_days=25)
    gap = m15.index[22 * 96 + 40]          # day 22, 10:00 UTC
    with_gap = m15.drop(index=gap)
    out = add_intraday_features(with_gap, make_daily())
    q = pd.Timedelta("15min")

    def nan_at(column, start, end):
        rows = out.loc[start:end, column]
        assert len(rows) > 0 and rows.isna().all(), column

    def valid_at(column, time):
        assert not np.isnan(out.loc[time, column]), (column, time)

    # Returns need the close exactly 1h / 4h / 24h earlier.
    for column, lag in [("ret_1h", "1h"), ("ret_4h", "4h"), ("ret_24h", "24h")]:
        nan_at(column, gap + pd.Timedelta(lag), gap + pd.Timedelta(lag))
        valid_at(column, gap + pd.Timedelta(lag) + q)
        valid_at(column, gap + q)

    # Realized volatility needs every 15-minute return in its window.
    for column, window in [("rv_4h", "4h"), ("rv_24h", "24h")]:
        nan_at(column, gap + q, gap + pd.Timedelta(window))
        valid_at(column, gap + pd.Timedelta(window) + q)
        valid_at(column, gap - q)

    # Relative volume needs today's 1h window and the same window on each of 20 past days.
    nan_at("rel_volume_1h", gap + q, gap + pd.Timedelta("45min"))
    valid_at("rel_volume_1h", gap + pd.Timedelta("1h"))
    valid_at("rel_volume_1h", gap - q)
    for days in [1, 2]:
        later = gap + pd.Timedelta(days=days)
        nan_at("rel_volume_1h", later, later + pd.Timedelta("45min"))
        valid_at("rel_volume_1h", later + pd.Timedelta("1h"))


def test_hour_ny_across_daylight_saving_time():
    jan = int(pd.Timestamp("2024-01-15 13:00", tz="UTC").value // 10**6)
    jul = int(pd.Timestamp("2024-07-15 13:00", tz="UTC").value // 10**6)
    dst_start = int(pd.Timestamp("2024-03-10 06:00", tz="UTC").value // 10**6)
    m15 = pd.concat([
        make_m15(np.full(8, 100.0), start_ms=jan),
        make_m15(np.full(8, 100.0), start_ms=dst_start),
        make_m15(np.full(8, 100.0), start_ms=jul),
    ])
    out = add_intraday_features(m15, make_daily())

    def hours(ts):
        row = out.loc[pd.Timestamp(ts, tz="UTC")]
        return row["hour_utc"], row["hour_ny"]

    assert hours("2024-01-15 14:00") == (14, 9)
    assert hours("2024-07-15 14:00") == (14, 10)
    # US daylight saving time began 2024-03-10 at 07:00 UTC (02:00 EST -> 03:00 EDT).
    assert hours("2024-03-10 06:45") == (6, 1)
    assert hours("2024-03-10 07:00") == (7, 3)


def test_prev_day_labels_come_from_the_previous_utc_day():
    m15 = make_random_m15(n_days=4)
    daily = make_daily(n_days=4)
    out = add_intraday_features(m15, daily)

    day0 = out.index < pd.Timestamp("2024-01-02", tz="UTC")
    assert out.loc[day0, ["prev_day_trend", "prev_day_vol_regime"]].isna().all().all()
    for d in range(1, 4):
        day = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=d)
        rows = out.loc[day:day + pd.Timedelta("23h45min")]
        assert len(rows) == 96
        assert (rows["prev_day_trend"] == f"trend_{d - 1}").all()
        assert (rows["prev_day_vol_regime"] == f"vol_{d - 1}").all()


@pytest.mark.parametrize("cut", [100, 1000, 20 * 96 + 50, 21 * 96, 23 * 96 + 7])
def test_no_leakage_under_truncation(cut):
    m15 = make_random_m15(n_days=25)
    m15 = m15.drop(index=m15.index[[500, 1500, 22 * 96 + 3]])  # gaps should not matter
    daily = make_daily(n_days=25)
    full = add_intraday_features(m15, daily)

    truncated_m15 = m15.iloc[:cut]
    cut_time = truncated_m15.index[-1]
    # Only daily labels already final at the last candle's close: days before its day.
    truncated_daily = daily[daily.index < cut_time.normalize()]
    truncated = add_intraday_features(truncated_m15, truncated_daily)

    pd.testing.assert_frame_equal(truncated, full.iloc[:cut])


def test_inputs_are_not_modified():
    m15 = make_random_m15(n_days=3)
    daily = make_daily()
    m15_before, daily_before = m15.copy(deep=True), daily.copy(deep=True)

    add_intraday_features(m15, daily)

    pd.testing.assert_frame_equal(m15, m15_before)
    pd.testing.assert_frame_equal(daily, daily_before)
