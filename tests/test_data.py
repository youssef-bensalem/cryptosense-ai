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