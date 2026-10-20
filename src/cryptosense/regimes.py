import numpy as np
import pandas as pd

def add_regimes(df):

    """Add rule-based trend and volatility regime labels to a feature DataFrame.

    Every label for a given day uses only that day's data and earlier days,
    never later ones. The input DataFrame is not modified; a new DataFrame
    is returned.

    Requires the columns ``dist_sma_100``, ``mom_20`` and ``vol_20``, as
    produced by ``add_features``.

    Labels added
    ------------
    trend
        "up" when the close is above its 100-day average and 20-day momentum
        is positive; "down" when both are negative; "range" when they
        disagree (a pullback in an uptrend or a rebound in a downtrend).
    vol_regime
        "high" when 20-day volatility is above its median over the previous
        365 days, otherwise "low". This is relative to the past year, not an
        absolute level of volatility.

    Labels are NaN while their inputs are unavailable: the first 99 rows
    for ``trend`` and the first 384 rows for ``vol_regime``.

    Parameters
    ----------
    df : pandas.DataFrame
        Daily data with features, indexed by UTC date.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with the ``trend`` and ``vol_regime`` columns
        appended.
    """

    out = df.copy()

    up=(out["dist_sma_100"]>0) & (out["mom_20"]>0)
    down=(out["dist_sma_100"]<0) & (out["mom_20"]<0)
    labels=np.select([up, down], ["up", "down"], default="range")

    valid = out[["dist_sma_100", "mom_20"]].notna().all(axis=1)
    out["trend"] = pd.Series(labels, index=out.index).where(valid)

    vol_ref = out["vol_20"].rolling(365).median()

    labels = np.where(out["vol_20"] > vol_ref, "high", "low")
    valid = out["vol_20"].notna() & vol_ref.notna()
    out["vol_regime"] = pd.Series(labels, index=out.index).where(valid)

    return out