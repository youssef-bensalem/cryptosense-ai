"""CryptoSense AI dashboard. Run from the project root with: streamlit run app/main.py"""

from pathlib import Path

import pandas as pd
import numpy as np
import ccxt
import plotly.express as px
import streamlit as st

from cryptosense.data import load_processed, refresh_symbol, symbol_to_slug
from cryptosense.features import add_features
from cryptosense.regimes import add_regimes
from cryptosense.setups import SETUPS, explain_setup, setup_flag
from cryptosense.volatility import scaled_vol_forecast
from cryptosense.backtest import run_backtest, summarize_backtest
from cryptosense.explain import (
    describe_setup, describe_trend, describe_volatility, evidence_meaning, evidence_strength,
    position_size, stop_level, suggested_stop, typical_move_range,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

REPLICATION_NOTES = {
    "ETH/USDT": (
        "Setups were designed on BTC/USDT. On ETH/USDT (notebook 08), neither setup showed an "
        "advantage over random trades taken during uptrends."
    ),
    "SOL/USDT": (
        "Setups were designed on BTC/USDT. On SOL/USDT (notebook 08), the continuation setup beat "
        "an average day of the same year in every year since 2021, but the advantage has shrunk "
        "from about +10% to about +2% per 5 days in recent years. The pullback setup underperformed."
    ),
}

@st.cache_data
def load_symbol(symbol):
    """Processed data with features and regimes, cached between reruns."""
    return add_regimes(add_features(load_processed(symbol, DATA_DIR)))

COST_PER_SIDE = 0.0015  # 0.10% fee + 0.05% slippage, as assumed in Phase 6


@st.cache_data
def run_strategies(symbol, start):
    """Backtest the continuation setup (5- and 10-day holds) and buy-and-hold from `start`."""
    data = load_symbol(symbol).loc[start:].dropna(subset=["dist_sma_100", "rel_volume_20", "er_20"])
    summaries, curves = {}, {}

    for hold in [5, 10]:
        trades, equity = run_backtest(data, setup_flag(data, "continuation"), hold, COST_PER_SIDE)
        summaries[f"continuation {hold}d"] = summarize_backtest(trades, equity)
        curves[f"continuation {hold}d"] = equity

    hold_all = pd.Series(False, index=data.index)
    hold_all.iloc[0] = True
    trades, equity = run_backtest(data, hold_all, len(data) - 2, COST_PER_SIDE)
    summary = summarize_backtest(trades, equity)
    summary["win_rate"] = summary["avg_trade"] = None
    summaries["buy and hold"] = summary
    curves["buy and hold"] = equity

    return pd.DataFrame(summaries).T, pd.DataFrame(curves)

st.set_page_config(page_title="CryptoSense AI", layout="wide")
st.title("CryptoSense AI")
st.caption("Research and decision support only. Not financial advice; no trades are placed.")

symbol = st.sidebar.selectbox("Asset", WATCHLIST)

if st.sidebar.button("Refresh data from Binance"):
    try:
        with st.sidebar.status("Fetching and validating data..."):
            exchange = ccxt.binance()
            now_ms = exchange.milliseconds()
            results = [refresh_symbol(exchange, s, DATA_DIR, now_ms) for s in WATCHLIST]
    except Exception as error:
        st.sidebar.error(f"Refresh failed: {error}. Any symbols updated before the error were kept.")
    else:
        st.cache_data.clear()
        st.session_state["refresh_results"] = results
        st.rerun()

if "refresh_results" in st.session_state:
    for r in st.session_state["refresh_results"]:
        problems = "clean" if not r["problems"] else f"problems: {r['problems']}"
        st.sidebar.caption(f"{r['symbol']}: {r['rows']} days to {r['last_date']} ({problems})")

processed_file = DATA_DIR / "processed" / f"{symbol_to_slug(symbol)}_1d.csv"
if not processed_file.exists():
    st.info(f"No data for {symbol} yet. Click **Refresh data from Binance** in the sidebar to download it.")
    st.stop()

df = load_symbol(symbol)

last_date = df.index[-1]
age_days = (pd.Timestamp.now(tz="UTC").normalize() - last_date).days
st.sidebar.write(f"Last complete candle: **{last_date.date()}**")
st.sidebar.write(f"Data age: {age_days} day(s)")

summary, overview, conditions, setups, volatility, backtest = st.tabs(
    ["Summary", "Market overview", "Market conditions", "Setup research", "Volatility", "Backtesting"]
)

with summary:
    today = df.iloc[-1]
    close_today = today["close"]
    st.subheader(f"{symbol}: {describe_trend(today['trend'])}, {describe_volatility(today['vol_regime'])}")
    st.caption(f"Based on the last complete day, {last_date.date()}.")

    change_20d = np.exp(today["mom_20"]) - 1
    direction = "risen" if change_20d >= 0 else "fallen"
    if pd.isna(today["dist_sma_100"]):
        position = "there is not enough history yet to compare it with its 100-day average"
    elif today["dist_sma_100"] >= 0:
        position = "it is above its 100-day average"
    else:
        position = "it is below its 100-day average"
    st.write(f"Over the past 20 days the price has {direction} {abs(change_20d):.1%}, and {position}.")

    st.markdown("#### How much could the price move this week?")
    move, low, high = typical_move_range(close_today, scaled_vol_forecast(df, 5).iloc[-1], horizon=5)
    st.write(
        f"A typical move over the next 5 days is about ±{move:.1%}: from today's close of "
        f"{close_today:,.2f}, that is roughly {low:,.2f} to {high:,.2f} USDT."
    )
    st.caption("This is a typical move; larger moves do happen.")

    st.markdown("#### Patterns")
    if symbol in REPLICATION_NOTES:
        st.warning(REPLICATION_NOTES[symbol])
    for setup in SETUPS:
        e = explain_setup(df, last_date, setup)
        if e["flagged"]:
            have_stats = e["past_mean_return"] is not None and e["past_mean_se"] is not None
            gap = e["past_mean_return"] - e["base_same_trend_mean"] if have_stats else float("nan")
            se = e["past_mean_se"] if have_stats else float("nan")
            badge = evidence_strength(gap, se, e["past_episodes"])
            st.write(f"**Pattern found:** {describe_setup(setup)}")
            st.write(f"Evidence: **{badge}**. {evidence_meaning(badge)}")
        else:
            st.write(f"**Not present today:** {describe_setup(setup)}")

    st.markdown("#### Risk helper")
    col1, col2 = st.columns(2)
    account = col1.number_input("Account size (USDT)", min_value=1.0, value=1000.0, step=100.0)
    risk_pct = col2.number_input(
        "Share of the account you are willing to lose on one trade (%)",
        min_value=0.1, max_value=100.0, value=1.0, step=0.5,
    )
    stop_price, stop_fraction = stop_level(close_today, suggested_stop(move))
    invest = position_size(account, risk_pct / 100, stop_fraction)
    col1, col2 = st.columns(2)
    col1.metric("Suggested stop", f"{stop_price:,.2f} USDT", f"-{stop_fraction:.1%} below today's close",
                delta_color="off")
    col2.metric("Position size", f"{invest:,.2f} USDT")
    st.write(
        f"Placing the stop at twice the typical 5-day move, investing {invest:,.2f} USDT means that "
        f"hitting the stop loses about {account * risk_pct / 100:,.2f} USDT ({risk_pct:g}% of the account), "
        "before costs."
    )
    if invest > account:
        st.warning("This is more than the account size; it would require borrowing, which this tool does not assume.")
    st.caption("Prices can jump past a stop in fast markets.")

    st.divider()
    st.write("This summary describes market conditions and past evidence. It is not advice to buy or sell.")
    st.caption("The other tabs show the full technical details behind these numbers.")

with overview:
    close = df["close"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Last close (USDT)", f"{close.iloc[-1]:,.2f}")
    col2.metric("1-day change", f"{close.iloc[-1] / close.iloc[-2] - 1:+.2%}")
    col3.metric("30-day change", f"{close.iloc[-1] / close.iloc[-31] - 1:+.2%}")

    price_fig = px.line(df, y="close", log_y=True, title=f"{symbol} daily close (log scale)")
    st.plotly_chart(price_fig, width="stretch")

    volume_fig = px.bar(df, y="volume", title=f"{symbol} daily volume (base asset units)")
    st.plotly_chart(volume_fig, width="stretch")

with conditions:
    today = df.iloc[-1]
    st.subheader(f"Current conditions ({last_date.date()})")

    col1, col2 = st.columns(2)
    col1.metric("Trend regime", str(today["trend"]))
    col2.metric("Volatility regime", str(today["vol_regime"]))
    st.caption(
        "Trend: up = above the 100-day average with positive 20-day momentum; "
        "down = both negative; range = they disagree. "
        "Volatility: high/low relative to its median over the previous 365 days."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("20-day momentum", f"{np.exp(today['mom_20']) - 1:+.1%}")
    col2.metric("Distance from 100-day average", f"{today['dist_sma_100']:+.1%}")
    col3.metric("Volatility (20d, annualised)", f"{today['vol_20'] * np.sqrt(365):.0%}")
    col4.metric("Efficiency ratio (20d)", f"{today['er_20']:.2f}")

    regime_fig = px.scatter(
        df.dropna(subset=["trend"]), y="close", color="trend", log_y=True,
        color_discrete_map={"up": "#2ca02c", "down": "#d62728", "range": "#7f7f7f"},
        title=f"{symbol} trend regimes",
    )
    regime_fig.update_traces(marker={"size": 3})
    st.plotly_chart(regime_fig, width="stretch")

    shares = df["trend"].value_counts(normalize=True).rename("share of days")
    st.dataframe(shares.to_frame().style.format("{:.1%}"))

with setups:
    st.caption(
        "Setups are research candidates, not buy or sell signals. Rules were pre-registered "
        "and evaluated on BTC/USDT; historical statistics describe the past and are not a forecast."
    )
    if symbol in REPLICATION_NOTES:
        st.warning(REPLICATION_NOTES[symbol])

    for setup in SETUPS:
        e = explain_setup(df, last_date, setup)
        status = "FLAGGED" if e["flagged"] else "not flagged"
        st.subheader(f"{setup.capitalize()} setup: {status} on {e['date']}")

        conditions_table = pd.DataFrame(e["conditions"])
        conditions_table["met"] = conditions_table["met"].map({True: "yes", False: "no"})
        st.dataframe(conditions_table, hide_index=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Typical 5-day move", f"±{e['typical_move']:.1%}")
        col2.metric("Past episodes", e["past_episodes"])
        if e["past_episodes"] > 1:
            col3.metric(
                "Past avg 5-day return",
                f"{e['past_mean_return']:+.2%}",
                help=f"Standard error ±{e['past_mean_se']:.2%}",
            )
            col4.metric("Avg day in same trend regime", f"{e['base_same_trend_mean']:+.2%}")
            gap = e["past_mean_return"] - e["base_same_trend_mean"]
            st.caption(
                f"Difference from an average {e['trend']}-trend day: {gap:+.2%} "
                f"({gap / e['past_mean_se']:+.1f} standard errors). "
                "Below about 2 standard errors, the difference could be chance."
            )

        flags = setup_flag(df, setup)
        starts = flags & ~flags.shift(fill_value=False)
        recent = starts[starts].index[-5:]
        st.write("Most recent episode starts: " + ", ".join(str(d.date()) for d in recent))

with volatility:
    HORIZON = 5
    forecast = scaled_vol_forecast(df, HORIZON)
    latest = forecast.iloc[-1]

    st.subheader(f"Volatility forecast for the next {HORIZON} days ({last_date.date()})")
    col1, col2, col3 = st.columns(3)
    col1.metric("Daily volatility", f"{latest:.2%}")
    col2.metric(f"Typical {HORIZON}-day move", f"±{latest * np.sqrt(HORIZON):.1%}")
    col3.metric("Annualised", f"{latest * np.sqrt(365):.0%}")
    st.caption(
        "Forecast = 20-day volatility × a scale factor learned only from past, completed outcomes. "
        "In Phase 4 (BTC), this scaled forecast beat raw 20-day volatility by 8–11% in every "
        "validation and test year; a Random Forest performed comparably and was not adopted."
    )

    realized = np.sqrt((df["log_return"] ** 2).rolling(HORIZON).mean()).shift(-HORIZON)
    comparison = pd.DataFrame({
        "forecast": forecast,
        f"realized next {HORIZON} days": realized,
    }).loc[last_date - pd.Timedelta(days=730):] * np.sqrt(365)
    vol_fig = px.line(comparison, title="Forecast vs realized volatility, last 2 years (annualised)")
    vol_fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(vol_fig, width="stretch")
    st.caption(
        f"Realized volatility is only known {HORIZON} days later, so the most recent "
        "days show the forecast alone."
    )

with backtest:
    st.caption(
        "Continuation setup, long only: enter at the next day's open after a signal, hold 5 or 10 days, "
        "one position at a time, all-in, 0.30% round-trip costs. Past performance does not "
        "guarantee future results."
    )
    if symbol in REPLICATION_NOTES:
        st.warning(REPLICATION_NOTES[symbol])
        
    period = st.radio("Period", ["Full history", "Held-out test (2024 onward)"], horizontal=True)
    start = "2024-01-01" if period.startswith("Held-out") else None
    summary_table, curves = run_strategies(symbol, start)

    percent_columns = ["total_return", "annual_volatility", "max_drawdown",
                       "time_in_market", "win_rate", "avg_trade"]
    st.dataframe(
        summary_table.style.format({c: "{:.1%}" for c in percent_columns} | {"trades": "{:.0f}"},
                                   na_rep="—")
    )

    equity_fig = px.line(curves, log_y=True, title="Account value (start = 1.0, log scale)")
    st.plotly_chart(equity_fig, width="stretch")

    st.caption(
        "Phase 6 findings (BTC): the continuation strategy stayed profitable after costs on held-out "
        "data, with far smaller drawdowns and a better return per unit of risk than buy-and-hold, but "
        "a lower total return. Its advantage over random trades taken during uptrends is suggestive, "
        "not established; trade counts are small."
    )
