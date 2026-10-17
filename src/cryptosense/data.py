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