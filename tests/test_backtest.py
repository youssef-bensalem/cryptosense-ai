import numpy as np
import pandas as pd
import pytest

from cryptosense.backtest import find_one_day_trades, find_trades, run_backtest, summarize_backtest


def make_prices(opens):
    """Daily DataFrame with the given open prices (close = next open)."""
    index = pd.date_range("2024-01-01", periods=len(opens), freq="D", tz="UTC", name="date")
    opens = pd.Series(opens, index=index, dtype=float)
    return pd.DataFrame({"open": opens, "close": opens.shift(-1).fillna(opens.iloc[-1])})


def make_signal(n, true_positions):
    signal = pd.Series(False, index=range(n))
    signal.iloc[true_positions] = True
    return signal


def test_entry_is_next_day_and_exit_after_hold():
    assert find_trades(make_signal(10, [2]), hold_days=3) == [(3, 6)]


def test_signals_during_an_open_trade_are_ignored():
    signal = make_signal(12, [2, 4, 6])
    # Trade from day 3 to day 6; the signal on day 4 is ignored;
    # the signal on day 6 (exit day) can start a new trade on day 7.
    assert find_trades(signal, hold_days=3) == [(3, 6), (7, 10)]


def test_no_trade_without_enough_data_to_exit():
    assert find_trades(make_signal(8, [5]), hold_days=3) == []


def test_costs_on_flat_prices():
    df = make_prices([100] * 10)
    signal = pd.Series(False, index=df.index)
    signal.iloc[2] = True

    trades, equity = run_backtest(df, signal, hold_days=3, cost_per_side=0.0015)

    expected = 0.9985 / 1.0015 - 1
    assert trades["gross_return"].iloc[0] == pytest.approx(0.0)
    assert trades["net_return"].iloc[0] == pytest.approx(expected)
    assert equity.iloc[-1] == pytest.approx(1 + expected)


def test_entry_uses_next_open_not_signal_close():
    opens = [100, 100, 100, 200, 200, 200, 400, 400]
    df = make_prices(opens)
    signal = pd.Series(False, index=df.index)
    signal.iloc[2] = True  # signal day's close (200) is already the next open

    trades, _ = run_backtest(df, signal, hold_days=3, cost_per_side=0.0)

    assert trades["entry_price"].iloc[0] == 200
    assert trades["exit_price"].iloc[0] == 400
    assert trades["gross_return"].iloc[0] == pytest.approx(1.0)


def test_final_equity_equals_compounded_trades():
    rng = np.random.default_rng(0)
    df = make_prices(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 300))))
    signal = pd.Series(rng.random(300) < 0.05, index=df.index)

    trades, equity = run_backtest(df, signal, hold_days=5)

    assert len(trades) > 5
    assert equity.iloc[-1] == pytest.approx((1 + trades["net_return"]).prod())

def test_summary_known_values():
    df = make_prices([100, 100, 100, 50, 50, 150, 150, 150])
    signal = pd.Series(False, index=df.index)
    signal.iloc[1] = True  # enter day 2 at 100, exit day 5 at 150

    trades, equity = run_backtest(df, signal, hold_days=3, cost_per_side=0.0)
    summary = summarize_backtest(trades, equity)

    assert summary["trades"] == 1
    assert summary["total_return"] == pytest.approx(0.5)
    assert summary["max_drawdown"] == pytest.approx(-0.5)   # 100 -> 50 while holding
    assert summary["win_rate"] == 1.0
    assert summary["time_in_market"] == pytest.approx(3 / 7)


def test_one_day_trades_one_per_signal_day_with_correct_positions():
    assert find_one_day_trades(make_signal(10, [2, 5])) == [(3, 4), (6, 7)]


def test_one_day_trades_keep_back_to_back_signals():
    # find_trades would ignore the signal on day 3; one-day trades keep both.
    signal = make_signal(10, [2, 3, 4])
    assert find_one_day_trades(signal) == [(3, 4), (4, 5), (5, 6)]
    assert find_trades(signal, hold_days=1) == [(3, 4), (5, 6)]


def test_one_day_trades_need_the_exit_day_in_the_data():
    # n = 6: a signal on day 3 exits on day 5 (the last row); signals on days 4 and 5 cannot exit.
    assert find_one_day_trades(make_signal(6, [3, 4, 5])) == [(4, 5)]
    assert find_one_day_trades(make_signal(6, [])) == []


def test_one_day_trades_match_flagged_positions_on_a_dated_series():
    rng = np.random.default_rng(0)
    signal = pd.Series(rng.random(200) < 0.3,
                       index=pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC"))
    trades = find_one_day_trades(signal)
    flagged = np.flatnonzero(signal.to_numpy())
    assert [entry - 1 for entry, _ in trades] == [t for t in flagged if t + 2 < len(signal)]
    assert all(exit_ == entry + 1 for entry, exit_ in trades)
