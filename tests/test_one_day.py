import numpy as np
import pandas as pd
import pytest

from cryptosense.data import candles_to_dataframe
from cryptosense.one_day import (
    ONE_DAY_SETUPS,
    RV_OPEN_THRESHOLDS,
    one_day_signals,
    us_open_measures,
)

CANDLE = pd.Timedelta("15min")
WINDOW_RETURNS = np.array([0.01, -0.02, 0.03, 0.0, 0.005, -0.01, 0.02, 0.01])


def utc(ts):
    return pd.Timestamp(ts, tz="UTC")


def make_m15(start, n_candles, returns=None):
    """15-minute candles from `start` (UTC) whose close-to-close log returns are `returns`."""
    returns = np.zeros(n_candles) if returns is None else returns
    close = 100 * np.exp(np.cumsum(returns))
    start_ms = int(utc(start).value // 10**6)
    candles = [[start_ms + i * 15 * 60 * 1000, c, c, c, c, 1.0] for i, c in enumerate(close)]
    return candles_to_dataframe(candles)


def day_with_window_returns(day, first_window_utc):
    """One UTC day of candles; the 8 window candles from `first_window_utc` carry WINDOW_RETURNS.

    Large returns are placed just outside the window (the 08:45 New York candle and the
    11:00 New York candle) so that a wrongly placed window gives different values.
    """
    index = pd.date_range(utc(day), periods=96, freq="15min")
    returns = pd.Series(0.0, index=index)
    first = utc(first_window_utc)
    returns[first - CANDLE] = 0.5          # 08:45 New York: its own return is outside the window
    returns[first + 8 * CANDLE] = -0.4     # 11:00 New York: outside the window
    returns.loc[first:first + 7 * CANDLE] = WINDOW_RETURNS
    return make_m15(day, 96, returns.to_numpy())


@pytest.mark.parametrize("day, first_window_utc", [
    ("2024-01-15", "2024-01-15 14:00"),   # standard time: 09:00 New York = 14:00 UTC
    ("2024-07-15", "2024-07-15 13:00"),   # daylight saving time: 09:00 New York = 13:00 UTC
])
def test_us_open_measures_known_values_both_sides_of_dst(day, first_window_utc):
    out = us_open_measures(day_with_window_returns(day, first_window_utc))
    assert out.loc[utc(day), "rv_open"] == pytest.approx(np.sqrt(np.sum(WINDOW_RETURNS ** 2)))
    assert out.loc[utc(day), "ret_open"] == pytest.approx(np.sum(WINDOW_RETURNS))


@pytest.mark.parametrize("drop, valid", [
    ("2024-01-15 13:45", False),   # 08:45 New York: needed by the first window return
    ("2024-01-15 14:00", False),   # first window candle
    ("2024-01-15 15:00", False),   # 10:00 New York
    ("2024-01-15 15:45", False),   # last window candle
    ("2024-01-15 16:00", True),    # 11:00 New York: after the window
    ("2024-01-15 13:30", True),    # 08:30 New York: before the window
])
def test_missing_candle_in_window_gives_nan(drop, valid):
    m15 = day_with_window_returns("2024-01-15", "2024-01-15 14:00").drop(index=utc(drop))
    row = us_open_measures(m15).loc[utc("2024-01-15")]
    assert row.notna().all() == valid
    if valid:
        assert row["rv_open"] == pytest.approx(np.sqrt(np.sum(WINDOW_RETURNS ** 2)))


@pytest.mark.parametrize("day, last_window_utc", [
    ("2024-01-15", "2024-01-15 15:45"),
    ("2024-07-15", "2024-07-15 14:45"),
])
def test_window_ends_before_the_daily_close(day, last_window_utc):
    rng = np.random.default_rng(0)
    m15 = make_m15(day, 96, rng.normal(0, 0.004, 96))
    full = us_open_measures(m15).loc[utc(day)]

    # Candles after the window (10:45 New York candle closes at 11:00) do not matter.
    cut = us_open_measures(m15.loc[:utc(last_window_utc)]).loc[utc(day)]
    pd.testing.assert_series_equal(cut, full)
    changed = m15.copy()
    after = changed.index > utc(last_window_utc)
    changed.loc[after, "close"] *= 1.5
    pd.testing.assert_series_equal(us_open_measures(changed).loc[utc(day)], full)
    assert utc(last_window_utc) + CANDLE < utc(day) + pd.Timedelta(days=1)


# --- Signals ---

def make_inputs(rows):
    """Daily inputs and US-open measures from rows of (date, trend, log_return, rv_open, ret_open)."""
    index = pd.DatetimeIndex([utc(r[0]) for r in rows]).as_unit("ms")
    daily = pd.DataFrame({"trend": [r[1] for r in rows], "log_return": [r[2] for r in rows]}, index=index)
    measures = pd.DataFrame({"rv_open": [r[3] for r in rows], "ret_open": [r[4] for r in rows]}, index=index)
    return daily, measures


def test_rules_are_the_approved_ones():
    assert RV_OPEN_THRESHOLDS == {"BTC/USDT": 0.010688, "SOL/USDT": 0.021055}
    assert [(f, op) for f, op, _ in ONE_DAY_SETUPS["D1"]] == [("trend", "=="), ("rv_open", ">"), ("ret_open", ">")]
    assert [(f, op) for f, op, _ in ONE_DAY_SETUPS["D2"]] == [("rv_open", ">"), ("ret_open", "<"), ("log_return", ">")]


# 2024-01-15 is a Monday, so its entry day (Tuesday) is allowed.
@pytest.mark.parametrize("trend, log_return, rv, ret, btc, sol", [
    # D1: up trend, active open, rising open
    ("up", -0.01, 0.015, 0.002, {"D1": True, "D2": False}, {"D1": False, "D2": False}),
    ("up", -0.01, 0.025, 0.002, {"D1": True, "D2": False}, {"D1": True, "D2": False}),
    ("range", -0.01, 0.025, 0.002, {"D1": False, "D2": False}, {"D1": False, "D2": False}),
    ("up", -0.01, 0.010688, 0.002, {"D1": False, "D2": False}, {"D1": False, "D2": False}),  # equal: not above
    ("up", -0.01, 0.025, 0.0, {"D1": False, "D2": False}, {"D1": False, "D2": False}),
    # D2: active open that fell, day closed up (any trend)
    ("down", 0.01, 0.015, -0.002, {"D1": False, "D2": True}, {"D1": False, "D2": False}),
    ("down", 0.01, 0.022, -0.002, {"D1": False, "D2": True}, {"D1": False, "D2": True}),
    ("up", 0.0, 0.022, -0.002, {"D1": False, "D2": False}, {"D1": False, "D2": False}),
    ("up", 0.01, 0.021055, -0.002, {"D1": False, "D2": True}, {"D1": False, "D2": False}),  # equal to SOL's
])
def test_flags_follow_the_approved_conditions_per_asset(trend, log_return, rv, ret, btc, sol):
    daily, measures = make_inputs([("2024-01-15", trend, log_return, rv, ret)])
    for symbol, expected in [("BTC/USDT", btc), ("SOL/USDT", sol)]:
        flags = one_day_signals(daily, measures, symbol).iloc[0]
        assert flags.to_dict() == expected, symbol


def test_saturday_entries_are_never_signalled():
    # Every day satisfies D1 (and the D2 row variant satisfies D2); Fridays enter on Saturday.
    days = pd.date_range("2024-01-01", "2024-01-31", freq="D")
    d1_rows = [(str(d.date()), "up", -0.01, 0.05, 0.01) for d in days]
    d2_rows = [(str(d.date()), "down", 0.01, 0.05, -0.01) for d in days]
    for rows, name in [(d1_rows, "D1"), (d2_rows, "D2")]:
        daily, measures = make_inputs(rows)
        flags = one_day_signals(daily, measures, "BTC/USDT")[name]
        fridays = daily.index.day_name() == "Friday"
        assert not flags[fridays].any()
        assert flags[~fridays].all()


def test_missing_inputs_mean_no_signal():
    daily, measures = make_inputs([
        ("2024-01-15", None, -0.01, 0.05, 0.01),       # no trend label: D1 cannot fire
        ("2024-01-16", "up", -0.01, np.nan, 0.01),     # incomplete US-open window
        ("2024-01-17", "down", np.nan, 0.05, -0.01),   # no log_return: D2 cannot fire
    ])
    flags = one_day_signals(daily, measures, "BTC/USDT")
    assert not flags.any().any()
    # A day absent from the measures entirely does not signal either.
    flags = one_day_signals(daily, measures.iloc[:0], "BTC/USDT")
    assert not flags.any().any()


def test_unknown_asset_raises():
    daily, measures = make_inputs([("2024-01-15", "up", -0.01, 0.05, 0.01)])
    with pytest.raises(ValueError):
        one_day_signals(daily, measures, "ETH/USDT")


@pytest.mark.parametrize("cut_day", ["2024-01-03", "2024-01-12", "2024-01-20", "2024-01-29"])
def test_no_leakage_under_truncation(cut_day):
    rng = np.random.default_rng(1)
    n_days = 30
    m15 = make_m15("2024-01-01", n_days * 96, rng.normal(0, 0.004, n_days * 96))
    days = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC").as_unit("ms")
    daily = pd.DataFrame({"trend": rng.choice(["up", "down", "range"], n_days),
                          "log_return": rng.normal(0, 0.02, n_days)}, index=days)
    full = one_day_signals(daily, us_open_measures(m15), "BTC/USDT")
    assert full["D1"].any() and full["D2"].any()

    # Everything known at the close of the cut day: daily rows up to it, candles opening before its end.
    cut = utc(cut_day)
    cut_m15 = m15[m15.index < cut + pd.Timedelta(days=1)]
    truncated = one_day_signals(daily.loc[:cut], us_open_measures(cut_m15), "BTC/USDT")
    pd.testing.assert_frame_equal(truncated, full.loc[:cut])
