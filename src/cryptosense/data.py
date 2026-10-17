import pandas as pd
DAY_MS = 86_400_000

def fetch_ohlcv_history(exchange,
                         symbol, 
                         timeframe="1d", 
                         start="2017-01-01T00:00:00Z", 
                         limit=1000):
    
    """Fetch the full candle history for a symbol, paginating as needed.

    Parameters
    ----------
    exchange : ccxt.Exchange
        An exchange object created by the caller, e.g. ``ccxt.binance()``.
    symbol : str
        Market symbol in CCXT format, e.g. ``"BTC/USDT"``.
    timeframe : str, default "1d"
        Candle size, e.g. ``"1d"`` or ``"4h"``.
    start : str, default "2017-01-01T00:00:00Z"
        ISO 8601 UTC date to start from. If it is earlier than the market's
        first candle, the exchange returns candles from its first available one.
    limit : int, default 1000
        Candles per request. Must not exceed the exchange's maximum
        (1000 for Binance), or pagination stops after the first request.

    Returns
    -------
    list of list
        Candles as ``[timestamp_ms, open, high, low, close, volume]``, oldest
        first. Includes the current, still-forming candle.
    """
        
    all_candles = []
    since=exchange.parse8601(start)
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if len(batch) == 0:
            break
        all_candles.extend(batch)
        if len(batch) < limit:
            break
        since = batch[-1][0] + 1

    return all_candles

def candles_to_dataframe(candles):

    """Convert a list of OHLCV candles into a DataFrame indexed by UTC date.

    Parameters
    ----------
    candles : list of list
        Candles as ``[timestamp_ms, open, high, low, close, volume]``,
        as returned by ``fetch_ohlcv_history``.

    Returns
    -------
    pandas.DataFrame
        One row per candle with columns ``timestamp``, ``open``, ``high``,
        ``low``, ``close`` and ``volume``, indexed by a UTC ``DatetimeIndex``
        named ``date``.

    Raises
    ------
    ValueError
        If ``candles`` is empty.
    """

    if len(candles) == 0:
        raise ValueError("No candles to convert: the candle list is empty.")
    
    columns=["timestamp", "open", "high", "low", "close", "volume"]
    df=pd.DataFrame(candles, columns=columns)

    df["date"]=pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df=df.set_index("date")
    return df


def validate_ohlcv(df):

    """Run data-quality checks on a daily OHLCV DataFrame and report the results.

    Assumes daily candles: gaps are measured against one day (``DAY_MS``).
    The DataFrame is not modified; problems are counted, not fixed.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``candles_to_dataframe``.

    Returns
    -------
    dict
        One entry per check. ``"sorted"`` is a bool; every other value is
        an int count, where 0 means no problem found.
    """

    price_cols = ["open", "high", "low", "close"]

    # 1. Duplicate timestamps
    duplicate_dates = int(df.index.duplicated().sum())

    # 2. Chronological order
    is_sorted = df.index.is_monotonic_increasing

    # 3. Gaps
    diffs = df["timestamp"].diff()
    gaps = diffs[diffs > DAY_MS]
    missing_days = int((gaps / DAY_MS - 1).sum())

    # 4. Missing values
    missing_values = int(df.isna().sum().sum())

    # 5. Invalid values
    non_positive_prices = int((df[price_cols] <= 0).sum().sum())
    negative_volume = int((df["volume"] < 0).sum())
    zero_volume_days= int((df["volume"] == 0).sum())

    # 6. Candle consistency
    bad_high = df["high"] < df[["open", "close"]].max(axis=1)
    bad_low = df["low"] > df[["open", "close"]].min(axis=1)
    high_below_open_close = int(bad_high.sum())
    low_above_open_close = int(bad_low.sum())

    return {
        "duplicate_dates": duplicate_dates,
        "sorted": is_sorted,
        "gaps": len(gaps),
        "missing_days": missing_days,
        "missing_values": missing_values,
        "non_positive_prices": non_positive_prices,
        "negative_volume": negative_volume,
        "zero_volume_days": zero_volume_days,
        "high_below_open_close": high_below_open_close,
        "low_above_open_close": low_above_open_close,
    }



def drop_incomplete_candles(df, now_ms):

    """Remove candles whose period has not finished yet.

    Assumes daily candles: a candle is complete once its full day
    (``timestamp + DAY_MS``) has passed.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``candles_to_dataframe``.
    now_ms : int
        Current time in milliseconds since the epoch (UTC), e.g. from
        ``exchange.milliseconds()``. Passed in so tests can use a fixed time.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing only complete candles.
    """
        
    is_complete = df["timestamp"] + DAY_MS <= now_ms

    return df[is_complete]