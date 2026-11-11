"""Simple, transparent backtester for fixed-holding-period, long-only setups.

Assumptions (Phase 6): a signal at a day's close enters at the next day's
open; the position is held for a fixed number of days and sold at that
day's open; one position at a time, all-in, no leverage; signals arriving
while a trade is open are ignored; costs are charged on both entry and exit.
"""

import numpy as np
import pandas as pd

# Fee plus slippage charged on each buy and each sell: 0.10% fee + 0.05% slippage.
# The single source of the project's cost assumption; import it instead of retyping it.
COST_PER_SIDE = 0.0015


def find_trades(signal, hold_days):
    """Turn a daily signal into non-overlapping trades.

    A signal at position ``t`` enters at ``t + 1`` and exits at
    ``t + 1 + hold_days``. Signals that arrive while a trade is open are
    ignored; a signal on the exit day may start the next trade. Signals too
    close to the end of the data to complete a trade are skipped.

    Parameters
    ----------
    signal : pandas.Series of bool
        True on days the setup is flagged at the close.
    hold_days : int
        Number of days each trade is held.

    Returns
    -------
    list of tuple of int
        ``(entry_position, exit_position)`` pairs, as row positions.
    """
    trades = []
    flags = signal.to_numpy()
    n = len(flags)
    next_allowed = 0

    for t in range(n):
        if t < next_allowed or not flags[t]:
            continue
        entry = t + 1
        exit_ = entry + hold_days
        if exit_ >= n:
            break
        trades.append((entry, exit_))
        next_allowed = exit_

    return trades


def find_one_day_trades(signal):
    """Turn a daily signal into one-day trades, one per signal day.

    A signal at position ``t`` enters at ``t + 1`` and exits at ``t + 2``.
    Unlike ``find_trades``, signals on consecutive days are all kept: a
    trade exits at the open where the next one enters, so one-day trades
    never overlap. Each trade is a separate round trip and pays costs on
    both entry and exit. Signals whose exit would fall beyond the data
    are skipped.

    Parameters
    ----------
    signal : pandas.Series of bool
        True on days the setup is flagged at the close.

    Returns
    -------
    list of tuple of int
        ``(entry_position, exit_position)`` pairs, as row positions.
    """
    flags = signal.to_numpy()
    n = len(flags)
    return [(t + 1, t + 2) for t in range(n) if flags[t] and t + 2 < n]


def run_backtest(df, signal, hold_days, cost_per_side=COST_PER_SIDE):
    """Simulate trading a signal with open-price execution and costs.

    Parameters
    ----------
    df : pandas.DataFrame
        Daily data with an ``open`` column, indexed by date.
    signal : pandas.Series of bool
        True on days the setup is flagged at the close, aligned with ``df``.
    hold_days : int
        Number of days each trade is held.
    cost_per_side : float, default ``COST_PER_SIDE`` (0.0015)
        Fee plus slippage charged on each buy and each sell
        (0.0015 = 0.10% fee + 0.05% slippage).

    Returns
    -------
    trades : pandas.DataFrame
        One row per trade: entry and exit dates and prices, gross return,
        and net return after costs.
    equity : pandas.Series
        Account value at each day's open, starting at 1.0, marked to market
        during trades.
    """
    opens = df["open"]
    growth = pd.Series(1.0, index=df.index)
    rows = []

    for entry, exit_ in find_trades(signal, hold_days):
        p_in, p_out = opens.iloc[entry], opens.iloc[exit_]
        rows.append({
            "entry_date": df.index[entry],
            "exit_date": df.index[exit_],
            "entry_price": p_in,
            "exit_price": p_out,
            "gross_return": p_out / p_in - 1,
            "net_return": (p_out * (1 - cost_per_side)) / (p_in * (1 + cost_per_side)) - 1,
        })

        # Day-by-day price growth while holding, then costs at entry and exit.
        held = slice(entry + 1, exit_ + 1)
        growth.iloc[held] *= opens.iloc[held].to_numpy() / opens.iloc[entry:exit_].to_numpy()
        growth.iloc[entry] /= 1 + cost_per_side
        growth.iloc[exit_] *= 1 - cost_per_side

    columns = ["entry_date", "exit_date", "entry_price", "exit_price", "gross_return", "net_return"]
    trades = pd.DataFrame(rows, columns=columns)
    equity = growth.cumprod()
    return trades, equity


def summarize_backtest(trades, equity, days_per_year=365):
    """Summary metrics for a backtest.

    Returns a dict with total return, annualized return (compound growth per
    year over the whole period), annualized volatility of daily equity
    changes, return per unit of risk (annualized return divided by annualized
    volatility; None when either is unavailable), maximum drawdown, number of
    trades, share of days in the market, win rate and average net return per
    trade.
    """
    daily = equity.pct_change().dropna()
    held_days = sum((t.exit_date - t.entry_date).days for t in trades.itertuples())
    total_days = (equity.index[-1] - equity.index[0]).days
    growth = float(equity.iloc[-1] / equity.iloc[0])
    annual_return = growth ** (days_per_year / total_days) - 1 if total_days else None
    annual_volatility = float(daily.std() * np.sqrt(days_per_year))
    has_risk = annual_return is not None and annual_volatility > 0

    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "return_per_risk": annual_return / annual_volatility if has_risk else None,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "trades": int(len(trades)),
        "time_in_market": held_days / total_days if total_days else 0.0,
        "win_rate": float((trades["net_return"] > 0).mean()) if len(trades) else None,
        "avg_trade": float(trades["net_return"].mean()) if len(trades) else None,
    }