import numpy as np
import pandas as pd


def add_features(df):
    """Add the Phase 2 feature columns to a daily OHLCV DataFrame.

    Every feature for a given day uses only that day's data and earlier
    days, never later ones. The input DataFrame is not modified; a new
    DataFrame is returned.

    Assumes daily candles with ``close`` and ``volume`` columns, such as
    the output of ``load_processed``.

    Features added
    --------------
    log_return
        Log return from the previous close to today's close.
    mom_5, mom_20
        Log return over the last 5 / 20 days (momentum).
    dist_sma_20, dist_sma_100
        Distance of the close from its 20 / 100-day simple moving
        average, as a fraction (0.05 means 5% above).
    vol_20, vol_60
        Standard deviation of daily log returns over 20 / 60 days
        (daily volatility, not annualised).
    rel_volume_20
        Today's volume divided by the average volume of the previous
        20 days (1.0 means normal activity).
    er_20
        Efficiency ratio over 20 days: net move divided by total
        distance travelled, between 0 (choppy) and 1 (steady trend).

    The first rows of each feature are NaN while its window fills up
    (up to 99 rows for ``dist_sma_100``).

    Parameters
    ----------
    df : pandas.DataFrame
        Daily OHLCV data indexed by UTC date.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with the nine feature columns appended.
    """

    out = df.copy()

    out["log_return"] = np.log(out["close"] / out["close"].shift(1))

    out["mom_5"] = np.log(out["close"] / out["close"].shift(5))
    out["mom_20"] = np.log(out["close"] / out["close"].shift(20))

    out["dist_sma_20"] = out["close"] / out["close"].rolling(20).mean() - 1
    out["dist_sma_100"] = out["close"] / out["close"].rolling(100).mean() - 1

    out["vol_20"] = out["log_return"].rolling(20).std()
    out["vol_60"] = out["log_return"].rolling(60).std()

    out["rel_volume_20"] = out["volume"] / out["volume"].shift(1).rolling(20).mean()

    out["er_20"] = out["mom_20"].abs() / out["log_return"].abs().rolling(20).sum()

    return out
