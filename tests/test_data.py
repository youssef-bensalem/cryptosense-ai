import numpy as np
import pandas as pd
import pytest

from cryptosense.data import (
    DAY_MS,
    candles_to_dataframe,
    drop_incomplete_candles,
    fetch_ohlcv_history,
    validate_ohlcv,
    symbol_to_slug,
    save_dataset,
    load_processed,
    refresh_symbol,
    timeframe_to_ms,
    update_symbol,
)

START_MS = 1_704_067_200_000

def make_candles(n):
    """Return n clean, consecutive daily candles starting 2024-01-01."""
    return [[START_MS + i * DAY_MS, 100.0, 110.0, 90.0, 105.0, 50.0] for i in range(n)]

class FakeExchange:
    """Stand-in for a CCXT exchange that serves candles from a list.

    Mimics the Binance behaviour fetch_ohlcv_history relies on: ``since`` is
    inclusive, candles come back oldest first, and each request returns at
    most ``max_limit`` candles.
    """

    def __init__(self, candles, max_limit=1000):
        self.candles = candles
        self.max_limit = max_limit
        self.requests = []  # the `since` value of every request, in order

    def parse8601(self, date_string):
        return pd.Timestamp(date_string).value // 1_000_000

    def fetch_ohlcv(self, symbol, timeframe="1m", since=None, limit=None):
        self.requests.append(since)
        matching = [c for c in self.candles if c[0] >= since]
        return matching[: min(limit, self.max_limit)]

def test_clean_data_reports_no_problems():
    df = candles_to_dataframe(make_candles(10))
    report = validate_ohlcv(df)

    assert report["sorted"] is True
    for check, result in report.items():
        if check != "sorted":
            assert result == 0, f"{check} reported {result} on clean data"


def test_detects_gap():
    candles = make_candles(10)
    del candles[4:6]  # remove two consecutive days
    report = validate_ohlcv(candles_to_dataframe(candles))

    assert report["gaps"] == 1
    assert report["missing_days"] == 2


def test_empty_candles_raise_error():
    with pytest.raises(ValueError):
        candles_to_dataframe([])


def test_detects_duplicate_date():
    candles = make_candles(10)
    candles.insert(3, candles[3][:])
    report = validate_ohlcv(candles_to_dataframe(candles))

    assert report["duplicate_dates"] == 1


def test_detects_missing_value():
    candles = make_candles(10)
    candles[3][4] = None
    report = validate_ohlcv(candles_to_dataframe(candles))

    assert report["missing_values"] == 1

def test_drop_incomplete_candles_at_boundary():
    candles = make_candles(3)
    df=candles_to_dataframe(candles)
    at_end = drop_incomplete_candles(df, START_MS + 3 * DAY_MS)
    just_before=drop_incomplete_candles(df, START_MS + 3 * DAY_MS-1)
    assert len(at_end) == 3
    assert len(just_before) == 2


def test_fetch_paginates_full_history():
    candles = make_candles(2500)
    exchange = FakeExchange(candles)

    result = fetch_ohlcv_history(exchange, "BTC/USDT")

    assert result == candles
    assert len(exchange.requests) == 3


def test_fetch_respects_limit():
    candles = make_candles(1200)
    exchange = FakeExchange(candles)

    result = fetch_ohlcv_history(exchange, "BTC/USDT", limit=500)

    assert result == candles
    assert len(exchange.requests) == 3

def test_fetch_exact_multiple_of_limit():
    candles = make_candles(2000)
    exchange=FakeExchange(candles)

    results = fetch_ohlcv_history(exchange, "BTC/USDT", limit=1000)

    assert results == candles
    assert len(exchange.requests) == 3


@pytest.mark.parametrize(
    "symbol, expected",
    [
        ("BTC/USDT", "btc_usdt"),
        ("ETH/USDT", "eth_usdt"),
        ("SOL/USDT", "sol_usdt"),
    ],
)
def test_symbol_to_slug(symbol, expected):
    assert symbol_to_slug(symbol) == expected


def test_save_dataset_file_names(tmp_path):
    candles = make_candles(3)
    raw_path, processed_path = save_dataset(
        candles, candles_to_dataframe(candles), "ETH/USDT", tmp_path
    )

    assert raw_path == tmp_path / "raw" / "eth_usdt_1d_raw.csv"
    assert processed_path == tmp_path / "processed" / "eth_usdt_1d.csv"


def test_raw_file_matches_fetched_candles(tmp_path):
    candles = make_candles(3)
    raw_path, _ = save_dataset(
        candles, candles_to_dataframe(candles), "BTC/USDT", tmp_path
    )

    expected = pd.DataFrame(
        candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    pd.testing.assert_frame_equal(pd.read_csv(raw_path), expected)


def test_save_and_load_round_trip(tmp_path):
    candles = make_candles(5)
    df = candles_to_dataframe(candles)

    raw_path, processed_path = save_dataset(candles, df, "BTC/USDT", tmp_path)
    loaded = load_processed("BTC/USDT", tmp_path)

    assert raw_path.exists()
    assert processed_path.exists()
    pd.testing.assert_frame_equal(df, loaded)


def test_refresh_symbol_runs_the_full_pipeline(tmp_path):
    candles = make_candles(10)
    exchange = FakeExchange(candles)
    now_ms = START_MS + 9 * DAY_MS + 1  # day 10 has started but not finished

    result = refresh_symbol(exchange, "BTC/USDT", tmp_path, now_ms)

    assert result["rows"] == 9            # incomplete last candle dropped
    assert result["problems"] == {}
    loaded = load_processed("BTC/USDT", tmp_path)
    assert len(loaded) == 9
    assert (tmp_path / "raw" / "btc_usdt_1d_raw.csv").exists()


# Timeframes other than 1d

MINUTE_MS = 60_000
FIFTEEN_MIN_MS = 15 * MINUTE_MS


def make_15m_candles(n):
    """Return n clean, consecutive 15-minute candles starting 2024-01-01."""
    return [[START_MS + i * FIFTEEN_MIN_MS, 100.0, 110.0, 90.0, 105.0, 50.0] for i in range(n)]


@pytest.mark.parametrize(
    "timeframe, expected",
    [
        ("1m", MINUTE_MS),
        ("15m", FIFTEEN_MIN_MS),
        ("1h", 60 * MINUTE_MS),
        ("4h", 4 * 60 * MINUTE_MS),
        ("1d", DAY_MS),
        ("1w", 7 * DAY_MS),
    ],
)
def test_timeframe_to_ms(timeframe, expected):
    assert timeframe_to_ms(timeframe) == expected


@pytest.mark.parametrize("timeframe", ["", "d", "0h", "-1h", "1.5h", "15", "1M", "1y", "15min"])
def test_timeframe_to_ms_rejects_invalid(timeframe):
    with pytest.raises(ValueError):
        timeframe_to_ms(timeframe)


def test_clean_15m_data_reports_no_problems():
    report = validate_ohlcv(candles_to_dataframe(make_15m_candles(10)), "15m")

    assert report["sorted"] is True
    for check, result in report.items():
        if check != "sorted":
            assert result == 0, f"{check} reported {result} on clean 15m data"


def test_detects_gap_in_15m_data():
    candles = make_15m_candles(10)
    del candles[4:7]  # remove three consecutive 15-minute candles
    report = validate_ohlcv(candles_to_dataframe(candles), "15m")

    assert report["gaps"] == 1
    assert report["missing_days"] == 3  # counts missing candles


def test_15m_data_validated_as_daily_reports_no_gaps():
    # Shows that the timeframe matters: 15-minute spacing is never wider than a day.
    candles = make_15m_candles(10)
    del candles[4:7]
    report = validate_ohlcv(candles_to_dataframe(candles))

    assert report["gaps"] == 0


def test_drop_incomplete_15m_candles_at_boundary():
    df = candles_to_dataframe(make_15m_candles(3))
    at_end = drop_incomplete_candles(df, START_MS + 3 * FIFTEEN_MIN_MS, "15m")
    just_before = drop_incomplete_candles(df, START_MS + 3 * FIFTEEN_MIN_MS - 1, "15m")

    assert len(at_end) == 3
    assert len(just_before) == 2


def test_save_dataset_15m_file_names(tmp_path):
    candles = make_15m_candles(3)
    raw_path, processed_path = save_dataset(
        candles, candles_to_dataframe(candles), "BTC/USDT", tmp_path, timeframe="15m"
    )

    assert raw_path == tmp_path / "raw" / "btc_usdt_15m_raw.csv"
    assert processed_path == tmp_path / "processed" / "btc_usdt_15m.csv"


def test_15m_and_1d_files_are_kept_separate(tmp_path):
    daily = make_candles(5)
    intraday = make_15m_candles(8)
    daily_df = candles_to_dataframe(daily)
    intraday_df = candles_to_dataframe(intraday)

    save_dataset(daily, daily_df, "BTC/USDT", tmp_path)
    save_dataset(intraday, intraday_df, "BTC/USDT", tmp_path, timeframe="15m")

    pd.testing.assert_frame_equal(load_processed("BTC/USDT", tmp_path), daily_df)
    pd.testing.assert_frame_equal(
        load_processed("BTC/USDT", tmp_path, timeframe="15m"), intraday_df
    )


def test_refresh_symbol_15m_runs_the_full_pipeline(tmp_path):
    candles = make_15m_candles(10)
    exchange = FakeExchange(candles)
    now_ms = START_MS + 9 * FIFTEEN_MIN_MS + 1  # candle 10 has started but not finished

    result = refresh_symbol(exchange, "BTC/USDT", tmp_path, now_ms, timeframe="15m")

    assert result["rows"] == 9            # incomplete last candle dropped
    assert result["problems"] == {}
    assert len(load_processed("BTC/USDT", tmp_path, timeframe="15m")) == 9
    assert (tmp_path / "raw" / "btc_usdt_15m_raw.csv").exists()
    assert not (tmp_path / "processed" / "btc_usdt_1d.csv").exists()


# Incremental updates

def test_fetch_since_ms_overrides_start():
    candles = make_candles(30)
    exchange = FakeExchange(candles)

    result = fetch_ohlcv_history(exchange, "BTC/USDT", since_ms=START_MS + 5 * DAY_MS)

    assert result == candles[5:]
    assert exchange.requests == [START_MS + 5 * DAY_MS]


def make_varied_candles(n, step_ms, seed=0):
    """n consecutive candles with random, non-round prices and volumes (to test exact CSV round trips)."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.uniform(10, 1000, n)
    return [[START_MS + i * step_ms, float(o), float(h), float(lo), float(c), float(v)]
            for i, (o, h, lo, c, v) in enumerate(zip(open_, high, low, close, volume))]


def forming(candle):
    """The still-forming version of a candle: same open, partial range, close and volume."""
    t, o, h, lo, c, v = candle
    return [t, o, max(o, (o + c) / 2), min(o, (o + c) / 2), (o + c) / 2, v / 3]


def first_run(tmp_path, candles, n_seen, step_ms, timeframe):
    """Save a first refresh that saw n_seen candles, the last one still forming."""
    served = candles[:n_seen - 1] + [forming(candles[n_seen - 1])]
    now_ms = candles[n_seen - 1][0] + 1
    refresh_symbol(FakeExchange(served), "BTC/USDT", tmp_path, now_ms, timeframe)


def read_files(data_dir, timeframe):
    return [(data_dir / sub / f"btc_usdt_{timeframe}{suffix}.csv").read_bytes()
            for sub, suffix in [("raw", "_raw"), ("processed", "")]]


TIMEFRAMES = [("1d", DAY_MS), ("15m", FIFTEEN_MIN_MS)]


@pytest.mark.parametrize("timeframe, step_ms", TIMEFRAMES)
def test_update_equals_full_refresh(tmp_path, timeframe, step_ms):
    candles = make_varied_candles(60, step_ms)
    first_run(tmp_path / "update", candles, 40, step_ms, timeframe)
    now_ms = candles[-1][0] + 1                      # candle 60 is still forming
    served = candles[:-1] + [forming(candles[-1])]

    updated = update_symbol(FakeExchange(served), "BTC/USDT", tmp_path / "update", now_ms, timeframe)
    refreshed = refresh_symbol(FakeExchange(served), "BTC/USDT", tmp_path / "full", now_ms, timeframe)

    assert updated == {**refreshed, "new_candles": 21}   # candles 40-60
    assert read_files(tmp_path / "update", timeframe) == read_files(tmp_path / "full", timeframe)
    pd.testing.assert_frame_equal(load_processed("BTC/USDT", tmp_path / "update", timeframe),
                                  candles_to_dataframe(candles[:-1]))


@pytest.mark.parametrize("timeframe, step_ms", TIMEFRAMES)
def test_update_replaces_incomplete_candle_without_duplicates(tmp_path, timeframe, step_ms):
    candles = make_varied_candles(60, step_ms)
    first_run(tmp_path, candles, 40, step_ms, timeframe)
    boundary = candles[39][0]
    assert boundary not in set(load_processed("BTC/USDT", tmp_path, timeframe)["timestamp"])

    exchange = FakeExchange(candles)
    update_symbol(exchange, "BTC/USDT", tmp_path, candles[-1][0] + step_ms, timeframe)

    assert exchange.requests[0] == boundary
    loaded = load_processed("BTC/USDT", tmp_path, timeframe)
    assert loaded.loc[loaded["timestamp"] == boundary, ["open", "high", "low", "close", "volume"]].values.tolist()         == [candles[39][1:]]
    raw = pd.read_csv(tmp_path / "raw" / f"btc_usdt_{timeframe}_raw.csv")
    for timestamps in [loaded["timestamp"], raw["timestamp"]]:
        assert timestamps.is_unique and timestamps.is_monotonic_increasing
        assert list(timestamps) == [c[0] for c in candles]


@pytest.mark.parametrize("timeframe, step_ms", TIMEFRAMES)
def test_update_makes_fewer_requests_than_refresh(tmp_path, timeframe, step_ms):
    candles = make_candles(2500) if timeframe == "1d" else make_15m_candles(2500)
    first_run(tmp_path, candles, 2495, step_ms, timeframe)
    now_ms = candles[-1][0] + step_ms

    update_exchange, refresh_exchange = FakeExchange(candles), FakeExchange(candles)
    result = update_symbol(update_exchange, "BTC/USDT", tmp_path, now_ms, timeframe)
    refresh_symbol(refresh_exchange, "BTC/USDT", tmp_path / "full", now_ms, timeframe)

    assert result["new_candles"] == 6
    assert len(update_exchange.requests) == 1
    assert len(refresh_exchange.requests) == 3


@pytest.mark.parametrize("timeframe, step_ms", TIMEFRAMES)
def test_update_without_saved_files_equals_refresh(tmp_path, timeframe, step_ms):
    candles = make_varied_candles(30, step_ms)
    now_ms = candles[-1][0] + 1

    updated = update_symbol(FakeExchange(candles), "BTC/USDT", tmp_path / "update", now_ms, timeframe)
    refreshed = refresh_symbol(FakeExchange(candles), "BTC/USDT", tmp_path / "full", now_ms, timeframe)

    assert updated == {**refreshed, "new_candles": 30}
    assert read_files(tmp_path / "update", timeframe) == read_files(tmp_path / "full", timeframe)


def test_update_when_all_saved_candles_are_complete(tmp_path):
    candles = make_varied_candles(30, DAY_MS)
    refresh_symbol(FakeExchange(candles[:20]), "BTC/USDT", tmp_path, candles[20][0], "1d")  # all 20 complete

    exchange = FakeExchange(candles)
    result = update_symbol(exchange, "BTC/USDT", tmp_path, candles[-1][0] + DAY_MS, "1d")

    assert exchange.requests[0] == candles[19][0] + 1
    assert result["new_candles"] == 10
    pd.testing.assert_frame_equal(load_processed("BTC/USDT", tmp_path), candles_to_dataframe(candles))

