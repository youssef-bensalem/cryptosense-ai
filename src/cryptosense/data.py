import pandas as pd
from pathlib import Path
DAY_MS = 86_400_000

TIMEFRAME_UNIT_MS = {
    "m": 60_000,
    "h": 3_600_000,
    "d": DAY_MS,
    "w": 7 * DAY_MS,
}


def timeframe_to_ms(timeframe):

    """Convert a timeframe string such as ``"15m"`` or ``"1d"`` into milliseconds.

    A pure replacement for ``exchange.parse_timeframe()`` that needs no
    exchange object. Supported units are ``m`` (minutes), ``h`` (hours),
    ``d`` (days) and ``w`` (weeks). Months and years are rejected because
    their length varies.

    Parameters
    ----------
    timeframe : str
        A positive integer followed by a unit, e.g. ``"15m"``, ``"1h"``,
        ``"4h"`` or ``"1d"``.

    Returns
    -------
    int
        Duration of one candle in milliseconds.

    Raises
    ------
    ValueError
        If the timeframe is not a positive integer followed by a supported unit.
    """

    amount, unit = timeframe[:-1], timeframe[-1:]
    if unit not in TIMEFRAME_UNIT_MS or not amount.isdigit() or int(amount) == 0:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}: expected a positive integer "
            f"followed by one of {sorted(TIMEFRAME_UNIT_MS)}, e.g. '15m' or '1d'."
        )
    return int(amount) * TIMEFRAME_UNIT_MS[unit]


def fetch_ohlcv_history(exchange,
                         symbol, 
                         timeframe="1d", 
                         start="2017-01-01T00:00:00Z",
                         limit=1000,
                         since_ms=None):
    
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
    since_ms : int, optional
        Timestamp in milliseconds (UTC) to start from instead of ``start``.
        ``None`` (default) uses ``start``.

    Returns
    -------
    list of list
        Candles as ``[timestamp_ms, open, high, low, close, volume]``, oldest
        first. Includes the current, still-forming candle.
    """
        
    all_candles = []
    since=exchange.parse8601(start) if since_ms is None else since_ms
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


def validate_ohlcv(df, timeframe="1d"):

    """Run data-quality checks on an OHLCV DataFrame and report the results.

    Gaps are measured against the candle duration of ``timeframe``.
    The DataFrame is not modified; problems are counted, not fixed.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``candles_to_dataframe``.
    timeframe : str, default "1d"
        Candle size of ``df``, e.g. ``"1d"`` or ``"15m"``; see
        ``timeframe_to_ms``.

    Returns
    -------
    dict
        One entry per check. ``"sorted"`` is a bool; every other value is
        an int count, where 0 means no problem found. The keys keep their
        daily names for every timeframe: ``"missing_days"`` counts missing
        candles and ``"zero_volume_days"`` counts zero-volume candles.
    """

    candle_ms = timeframe_to_ms(timeframe)

    price_cols = ["open", "high", "low", "close"]

    # 1. Duplicate timestamps
    duplicate_dates = int(df.index.duplicated().sum())

    # 2. Chronological order
    is_sorted = df.index.is_monotonic_increasing

    # 3. Gaps
    diffs = df["timestamp"].diff()
    gaps = diffs[diffs > candle_ms]
    missing_days = int((gaps / candle_ms - 1).sum())

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



def drop_incomplete_candles(df, now_ms, timeframe="1d"):

    """Remove candles whose period has not finished yet.

    A candle is complete once its full period (``timestamp`` plus the
    candle duration of ``timeframe``) has passed.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``candles_to_dataframe``.
    now_ms : int
        Current time in milliseconds since the epoch (UTC), e.g. from
        ``exchange.milliseconds()``. Passed in so tests can use a fixed time.
    timeframe : str, default "1d"
        Candle size of ``df``, e.g. ``"1d"`` or ``"15m"``; see
        ``timeframe_to_ms``.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing only complete candles.
    """
        
    is_complete = df["timestamp"] + timeframe_to_ms(timeframe) <= now_ms

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


def save_dataset(raw_candles, df, symbol, data_dir, timeframe="1d"):

    """Save raw and processed data for one symbol as CSV files.

    Writes ``<data_dir>/raw/<slug>_<timeframe>_raw.csv`` (candles exactly as
    fetched) and ``<data_dir>/processed/<slug>_<timeframe>.csv`` (validated
    DataFrame, with its ``date`` index as the first column), e.g.
    ``btc_usdt_1d.csv`` or ``btc_usdt_15m.csv``. Folders are created if needed.

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
    timeframe : str, default "1d"
        Candle size of the data, used in the file names.

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

    raw_path=raw_dir / f"{slug}_{timeframe}_raw.csv"
    processed_path=processed_dir / f"{slug}_{timeframe}.csv"

    raw_df = pd.DataFrame(raw_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    raw_df.to_csv(raw_path, index=False)

    df.to_csv(processed_path)

    return raw_path, processed_path


def load_processed(symbol, data_dir, timeframe="1d"):

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
    timeframe : str, default "1d"
        Candle size of the file to load, the same one passed to ``save_dataset``.

    Returns
    -------
    pandas.DataFrame
        Processed data indexed by UTC date.
    """

    slug=symbol_to_slug(symbol)
    processed_path=data_dir / "processed" / f"{slug}_{timeframe}.csv"

    reloaded = pd.read_csv(processed_path)
    reloaded["date"] = pd.to_datetime(reloaded["timestamp"], unit="ms", utc=True)
    reloaded = reloaded.set_index("date")

    return reloaded


def refresh_symbol(exchange, symbol, data_dir, now_ms, timeframe="1d"):
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
    timeframe : str, default "1d"
        Candle size to fetch, validate and save, e.g. ``"1d"`` or ``"15m"``.

    Returns
    -------
    dict
        ``symbol``, number of processed ``rows``, ``last_date`` and the
        ``problems`` found by validation (an empty dict means clean data).
    """
    candles = fetch_ohlcv_history(exchange, symbol, timeframe=timeframe)
    return _process_and_save(candles, symbol, data_dir, now_ms, timeframe)


def _process_and_save(candles, symbol, data_dir, now_ms, timeframe):
    """Validate, drop incomplete candles, save, and summarise, as in ``refresh_symbol``."""
    df = candles_to_dataframe(candles)
    report = validate_ohlcv(df, timeframe)
    df = drop_incomplete_candles(df, now_ms, timeframe)
    save_dataset(candles, df, symbol, data_dir, timeframe)

    problems = {
        check: result
        for check, result in report.items()
        if (check == "sorted" and result is not True) or (check != "sorted" and result != 0)
    }
    return {"symbol": symbol, "rows": len(df), "last_date": df.index[-1].date(), "problems": problems}


RAW_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def update_symbol(exchange, symbol, data_dir, now_ms, timeframe="1d"):
    """Update one symbol's saved data by fetching only the candles not yet processed.

    Without a saved raw file this is exactly ``refresh_symbol``. Otherwise the
    saved raw candles are loaded and the first one missing from the saved
    processed file (normally the candle that was incomplete on the last run)
    sets the starting timestamp; if every raw candle is processed, fetching
    starts just after the last one. Candles from that timestamp on are fetched
    again, combined with the saved raw candles before it (a fetched candle
    replaces a saved one with the same timestamp), sorted, and run through the
    same pipeline as ``refresh_symbol``: validation of the whole combined
    history, dropping incomplete candles, and ``save_dataset``.

    Parameters
    ----------
    exchange : ccxt.Exchange
        Exchange object created by the caller.
    symbol : str
        Market symbol in CCXT format, e.g. ``"BTC/USDT"``.
    data_dir : pathlib.Path
        Root data folder, the same one used by ``refresh_symbol``.
    now_ms : int
        Current time in milliseconds (UTC), used to drop incomplete candles.
    timeframe : str, default "1d"
        Candle size to update, e.g. ``"1d"`` or ``"15m"``.

    Returns
    -------
    dict
        The ``refresh_symbol`` summary plus ``new_candles``, the number of
        candles fetched.
    """
    slug = symbol_to_slug(symbol)
    raw_path = data_dir / "raw" / f"{slug}_{timeframe}_raw.csv"
    if not raw_path.exists():
        candles = fetch_ohlcv_history(exchange, symbol, timeframe=timeframe)
        return {**_process_and_save(candles, symbol, data_dir, now_ms, timeframe), "new_candles": len(candles)}

    # round_trip parsing reads back exactly the floats that were written.
    raw = pd.read_csv(raw_path, float_precision="round_trip")
    saved = [list(row) for row in zip(*(raw[c].tolist() for c in RAW_COLUMNS))]

    processed_path = data_dir / "processed" / f"{slug}_{timeframe}.csv"
    processed = set(pd.read_csv(processed_path)["timestamp"]) if processed_path.exists() else set()
    unprocessed = [c[0] for c in saved if c[0] not in processed]
    since_ms = unprocessed[0] if unprocessed else saved[-1][0] + 1

    fetched = fetch_ohlcv_history(exchange, symbol, timeframe=timeframe, since_ms=since_ms)
    by_timestamp = {c[0]: c for c in saved if c[0] < since_ms}
    by_timestamp.update({c[0]: c for c in fetched})
    candles = [by_timestamp[t] for t in sorted(by_timestamp)]

    return {**_process_and_save(candles, symbol, data_dir, now_ms, timeframe), "new_candles": len(fetched)}
