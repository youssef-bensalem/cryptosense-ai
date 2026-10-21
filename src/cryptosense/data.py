import pandas as pd
from pathlib import Path
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


#funtions used to extend to sol and eth

def symbol_to_slug(symbol):

    """Convert a market symbol into a lowercase, file-name-safe slug.

    Example: ``"BTC/USDT"`` becomes ``"btc_usdt"``.

    Intended for spot symbols. Futures symbols such as ``"BTC/USDT:USDT"``
    contain ``":"``, which is not allowed in Windows file names, and are
    not handled.

    Parameters
    ----------
    symbol : str
        Market symbol in CCXT format.

    Returns
    -------
    str
        The slug.
    """

    new=symbol.replace("/", "_")
    new=new.lower()
    return new


def save_dataset(raw_candles, df, symbol, data_dir):

    """Save raw and processed data for one symbol as CSV files.

    Writes ``<data_dir>/raw/<slug>_1d_raw.csv`` (candles exactly as fetched)
    and ``<data_dir>/processed/<slug>_1d.csv`` (validated DataFrame, with its
    ``date`` index as the first column). Folders are created if needed.

    Parameters
    ----------
    raw_candles : list of list
        Candles as returned by ``fetch_ohlcv_history``.
    df : pandas.DataFrame
        Processed data, e.g. after ``drop_incomplete_candles``.
    symbol : str
        Market symbol in CCXT format, e.g. ``"BTC/USDT"``.
    data_dir : pathlib.Path
        Root data folder chosen by the caller.

    Returns
    -------
    tuple of pathlib.Path
        ``(raw_path, processed_path)``.
    """

    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    slug=symbol_to_slug(symbol)

    raw_path=raw_dir / f"{slug}_1d_raw.csv"
    processed_path=processed_dir / f"{slug}_1d.csv"

    raw_df = pd.DataFrame(raw_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    raw_df.to_csv(raw_path, index=False)

    df.to_csv(processed_path)

    return raw_path, processed_path


def load_processed(symbol, data_dir):

    """Load one symbol's processed CSV with the exact types it was saved with.

    The ``date`` index is rebuilt from the integer ``timestamp`` column rather
    than parsed from text, so the result has a ``datetime64[ms, UTC]`` index
    and matches the DataFrame originally passed to ``save_dataset``.

    Parameters
    ----------
    symbol : str
        Market symbol in CCXT format, e.g. ``"BTC/USDT"``.
    data_dir : pathlib.Path
        Root data folder, the same one passed to ``save_dataset``.

    Returns
    -------
    pandas.DataFrame
        Processed data indexed by UTC date.
    """

    slug=symbol_to_slug(symbol)
    processed_path=data_dir / "processed" / f"{slug}_1d.csv"

    reloaded = pd.read_csv(processed_path)
    reloaded["date"] = pd.to_datetime(reloaded["timestamp"], unit="ms", utc=True)
    reloaded = reloaded.set_index("date")

    return reloaded


def refresh_symbol(exchange, symbol, data_dir, now_ms):
    """Run the full data pipeline for one symbol and save the results.

    Fetches the full history, validates it, drops incomplete candles, and
    saves raw and processed files with ``save_dataset``.

    Parameters
    ----------
    exchange : ccxt.Exchange
        Exchange object created by the caller.
    symbol : str
        Market symbol in CCXT format, e.g. ``"BTC/USDT"``.
    data_dir : pathlib.Path
        Root data folder.
    now_ms : int
        Current time in milliseconds (UTC), used to drop incomplete candles.

    Returns
    -------
    dict
        ``symbol``, number of processed ``rows``, ``last_date`` and the
        ``problems`` found by validation (an empty dict means clean data).
    """
    candles = fetch_ohlcv_history(exchange, symbol)
    df = candles_to_dataframe(candles)
    report = validate_ohlcv(df)
    df = drop_incomplete_candles(df, now_ms)
    save_dataset(candles, df, symbol, data_dir)

    problems = {
        check: result
        for check, result in report.items()
        if (check == "sorted" and result is not True) or (check != "sorted" and result != 0)
    }
    return {"symbol": symbol, "rows": len(df), "last_date": df.index[-1].date(), "problems": problems}