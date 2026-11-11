"""CryptoSense AI dashboard. Run from the project root with: streamlit run app/main.py"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
import ccxt
import plotly.express as px
import streamlit as st

from cryptosense.data import load_processed, symbol_to_slug, update_symbol
from cryptosense.features import add_features
from cryptosense.regimes import add_regimes
from cryptosense.setups import SETUPS, explain_setup, setup_flag
from cryptosense.volatility import (
    HAR_LABEL, HAR_SYMBOLS, daily_realized_vol, dashboard_forecast, realized_vol_ahead, rms_ahead,
)
from cryptosense.backtest import COST_PER_SIDE
from cryptosense.explain import (
    describe_setup, describe_trend, describe_volatility, evidence_gap, evidence_meaning,
    position_size, setup_evidence, stop_level, suggested_stop, typical_move_range,
)
from cryptosense.context import derive_insights
from cryptosense.facts import build_facts
from cryptosense.one_day import us_open_measures
from cryptosense.assistant import ask, get_local_client
from cryptosense import service
from cryptosense.service import REPLICATION_NOTES, WATCHLIST
from cryptosense.news.coverage import format_status, load_collector_status

# The project's data folder; CRYPTOSENSE_DATA_DIR overrides it (used by the dashboard smoke test).
DATA_DIR = Path(os.environ.get("CRYPTOSENSE_DATA_DIR") or Path(__file__).resolve().parents[1] / "data")
CHART_RANGES = {"1 month": pd.DateOffset(months=1), "3 months": pd.DateOffset(months=3),
                "1 year": pd.DateOffset(years=1), "All": None}

@st.cache_data
def load_symbol(symbol):
    """Processed data with features and regimes, cached between reruns."""
    return add_regimes(add_features(load_processed(symbol, DATA_DIR)))


@st.cache_data
def load_m15(symbol):
    """15-minute candles, or None if they have not been downloaded yet; cached between reruns."""
    if not (DATA_DIR / "processed" / f"{symbol_to_slug(symbol)}_15m.csv").exists():
        return None
    return load_processed(symbol, DATA_DIR, timeframe="15m")


@st.cache_data
def load_forecast(symbol, horizon=5):
    """Dashboard volatility forecast over `horizon` days and its method label, cached between reruns."""
    return dashboard_forecast(symbol, load_symbol(symbol), load_m15(symbol), horizon=horizon)


@st.cache_data
def load_rv_open(symbol, date):
    """US-open realized volatility on `date` from 15-minute data, or None (ETH, or no data that day)."""
    m15 = load_m15(symbol) if symbol in HAR_SYMBOLS else None
    if m15 is None:
        return None
    rv_open = us_open_measures(m15)["rv_open"].get(date)
    return None if rv_open is None or pd.isna(rv_open) else float(rv_open)


@st.cache_data
def load_active_patterns(symbol):
    """Setups flagged on the latest day: name, plain-language description and evidence badge."""
    return service.active_patterns(DATA_DIR, symbol)


@st.cache_resource(ttl=60)
def local_client():
    """Local Ollama client, or None if Ollama is not running; checked again after a minute."""
    return get_local_client()


@st.cache_data
def load_realized_ahead(symbol):
    """Realized volatility over the next 5 days from 15-minute data, on the daily index (None without 15-minute data)."""
    m15 = load_m15(symbol)
    if m15 is None:
        return None
    return realized_vol_ahead(daily_realized_vol(m15), horizon=5).reindex(load_symbol(symbol).index)


@st.cache_data
def run_strategies(symbol, start):
    """Backtest the continuation setup (5- and 10-day holds) and buy-and-hold from `start`."""
    return service.run_strategies(DATA_DIR, symbol, start)


def risk_helper(close, typical_move, basis, key):
    """Show the risk helper: account inputs, suggested stop, position size and what hitting the stop costs.

    `typical_move` is the typical move over the holding period, `basis` names it in the explanation
    (for example "tomorrow's typical swing"), and `key` keeps this helper's inputs separate from the
    other tab's.
    """
    col1, col2 = st.columns(2)
    account = col1.number_input("Account size (USDT)", min_value=1.0, value=1000.0, step=100.0,
                                key=f"account_{key}")
    risk_pct = col2.number_input(
        "Share of the account you are willing to lose on one trade (%)",
        min_value=0.1, max_value=100.0, value=1.0, step=0.5, key=f"risk_pct_{key}",
    )
    stop_price, stop_fraction = stop_level(close, suggested_stop(typical_move))
    invest = position_size(account, risk_pct / 100, stop_fraction)
    col1, col2 = st.columns(2)
    col1.metric("Suggested stop", f"{stop_price:,.2f} USDT", f"-{stop_fraction:.1%} below today's close",
                delta_color="off")
    col2.metric("Position size", f"{invest:,.2f} USDT")
    st.write(
        f"Placing the stop at twice {basis}, investing {invest:,.2f} USDT means that "
        f"hitting the stop loses about {account * risk_pct / 100:,.2f} USDT "
        f"({risk_pct:g}% of the account), before costs."
    )
    if invest > account:
        st.warning("This is more than the account size; it would require borrowing, which this tool does not assume.")
    st.caption("Prices can jump past a stop in fast markets.")

st.set_page_config(page_title="CryptoSense AI", layout="wide")
st.title("CryptoSense AI")
st.caption("Research and decision support only. Not financial advice; no trades are placed.")

symbol = st.sidebar.selectbox("Asset", WATCHLIST)

if st.sidebar.button("Refresh data from Binance"):
    try:
        with st.sidebar.status("Fetching and validating data..."):
            exchange = ccxt.binance()
            now_ms = exchange.milliseconds()
            results = [{**update_symbol(exchange, s, DATA_DIR, now_ms), "unit": "days"} for s in WATCHLIST]
            for s in HAR_SYMBOLS:
                if not (DATA_DIR / "raw" / f"{symbol_to_slug(s)}_15m_raw.csv").exists():
                    st.write(f"Downloading 15-minute data for {s} for the first time; this takes a few minutes.")
                results.append({**update_symbol(exchange, s, DATA_DIR, now_ms, timeframe="15m"),
                                "unit": "15-minute candles"})
    except Exception as error:
        st.sidebar.error(f"Refresh failed: {error}. Any symbols updated before the error were kept.")
    else:
        st.cache_data.clear()
        service.clear_cache()
        st.session_state["refresh_results"] = results
        st.rerun()

if "refresh_results" in st.session_state:
    for r in st.session_state["refresh_results"]:
        problems = "clean" if not r["problems"] else f"problems: {r['problems']}"
        st.sidebar.caption(f"{r['symbol']}: {r['rows']} {r['unit']} to {r['last_date']} ({problems})")

processed_file = DATA_DIR / "processed" / f"{symbol_to_slug(symbol)}_1d.csv"
if not processed_file.exists():
    st.info(f"No data for {symbol} yet. Click **Refresh data from Binance** in the sidebar to download it.")
    st.stop()

df = load_symbol(symbol)
forecast, forecast_method = load_forecast(symbol)
is_har = forecast_method == HAR_LABEL
swing = "swing" if is_har else "move"
swing_long = "typical swing (including moves within the day)" if is_har else "typical move"

last_date = df.index[-1]
age_days = (pd.Timestamp.now(tz="UTC").normalize() - last_date).days
st.sidebar.write(f"Last complete candle: **{last_date.date()}**")
st.sidebar.write(f"Data age: {age_days} day(s)")

news_now = pd.Timestamp.now(tz="UTC").to_pydatetime()
news_status = load_collector_status(DATA_DIR, news_now)
with st.sidebar.expander(f"News collector: {news_status['days_to_spare']} day(s) to spare"):
    st.caption("Data-quality check for the news study (hours rule only; sentiment is not shown).")
    for line in format_status(news_status, news_now):
        st.caption(line)

chart_range = st.sidebar.segmented_control(
    "Chart range", list(CHART_RANGES), default="3 months", required=True,
    help="Period shown in the daily charts.",
)
range_offset = CHART_RANGES[chart_range]
range_start = None if range_offset is None else last_date - range_offset


def in_range(frame):
    """The rows of a daily frame inside the selected chart range."""
    return frame if range_start is None else frame.loc[range_start:]


# Facts and deductions for tomorrow, shared by "Today & tomorrow" and the assistant.
forecast_1d, forecast_1d_method = load_forecast(symbol, horizon=1)
facts = build_facts(df, symbol, forecast_1d.iloc[-1], forecast_1d_method,
                    rv_open=load_rv_open(symbol, last_date), patterns=load_active_patterns(symbol))
insights = derive_insights(facts)

today_tab, assistant_tab, summary, overview, conditions, setups, volatility, backtest = st.tabs(
    ["Today & tomorrow", "Ask the assistant", "Summary", "Market overview", "Market conditions",
     "Patterns (research)", "Volatility", "Strategy test results"]
)

with today_tab:
    close_today = df["close"].iloc[-1]
    st.subheader(f"{symbol}: what to expect tomorrow ({(last_date + pd.Timedelta(days=1)).date()})")
    st.caption(f"Based on the last complete day, {last_date.date()}. Method: {forecast_1d_method}.")

    swing_1d_word = "swing" if forecast_1d_method == HAR_LABEL else "move"
    if facts["swing_1d"] is None or facts["range_low"] is None:
        st.info("Not enough history to estimate tomorrow's typical swing.")
    else:
        swing_1d_name = ("typical swing (including moves within the day)"
                         if forecast_1d_method == HAR_LABEL else "typical move")
        col1, col2 = st.columns(2)
        col1.metric(f"Tomorrow's {swing_1d_name}", f"±{facts['swing_1d']:.1%}")
        col2.metric("Typical range", f"{facts['range_low']:,.2f} – {facts['range_high']:,.2f} USDT")
        st.caption(f"From today's close of {close_today:,.2f} USDT. This is a typical size of move, "
                   "not a direction; larger moves do happen.")

    st.markdown("#### What this means")
    for insight in sorted(insights, key=lambda i: i.kind != "warning"):
        if insight.kind == "warning":
            st.warning(insight.text, title=insight.title)
            st.caption(f"Evidence: {insight.evidence}")
        else:
            with st.container(border=True):
                st.markdown(f"**{insight.title}**")
                st.write(insight.text)
                st.caption(f"Evidence: {insight.evidence}")

    st.markdown("#### Risk helper")
    if facts["swing_1d"] is not None:
        risk_helper(close_today, facts["swing_1d"], f"tomorrow's typical {swing_1d_word}", "1d")

    st.markdown("#### Last 3 days, every 15 minutes")
    m15 = load_m15(symbol) if symbol in HAR_SYMBOLS else None
    if m15 is None:
        st.caption("No intraday data for this asset." if symbol not in HAR_SYMBOLS else
                   "No 15-minute data yet. Click **Refresh data from Binance** to download it.")
    else:
        recent_m15 = m15.loc[m15.index[-1] - pd.Timedelta(days=3):]
        intraday_fig = px.line(recent_m15, y="close", title=f"{symbol} 15-minute close, last 3 days (UTC)")
        st.plotly_chart(intraday_fig, width="stretch")

    st.divider()
    st.write("This describes conditions and risk. It is not advice to buy or sell.")

with assistant_tab:
    client = local_client()
    if client is None:
        st.info(
            "To use the free local assistant:\n"
            "1. Install Ollama from [ollama.com/download](https://ollama.com/download).\n"
            "2. In PowerShell, run `ollama pull llama3.2`.\n"
            "3. Make sure Ollama is running, then reload this page.",
            title="The assistant is not available",
        )
    else:
        st.caption("Answers use only the facts and deductions on the **Today & tomorrow** tab. "
                   "Information, not financial advice.")
        chats = st.session_state.setdefault("chats", {})
        history = chats.setdefault(symbol, [])
        for message in history:
            with st.chat_message(message["role"]):
                content = message["content"]
                st.markdown(content.replace("$", "\\$") if message["role"] == "assistant" else content)
        if question := st.chat_input(f"Ask about {symbol} today", submit_mode="disable"):
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask(question, facts, insights, client, history=history)
                st.markdown(answer.replace("$", "\\$"))
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]

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
    move, low, high = typical_move_range(close_today, forecast.iloc[-1], horizon=5)
    st.write(
        f"A {swing_long} over the next 5 days is about ±{move:.1%}: from today's close of "
        f"{close_today:,.2f}, that is roughly {low:,.2f} to {high:,.2f} USDT."
    )
    st.caption(f"This is a typical {swing}; larger moves do happen.")

    st.markdown("#### Patterns")
    if symbol in REPLICATION_NOTES:
        st.warning(REPLICATION_NOTES[symbol])
    for setup in SETUPS:
        e = explain_setup(df, last_date, setup)
        if e["flagged"]:
            badge = setup_evidence(e)
            st.write(f"**Pattern found:** {describe_setup(setup)}")
            st.write(f"Evidence: **{badge}**. {evidence_meaning(badge)}")
        else:
            st.write(f"**Not present today:** {describe_setup(setup)}")

    st.markdown("#### Risk helper")
    risk_helper(close_today, move, f"the typical 5-day {swing}", "5d")

    st.divider()
    st.write("This summary describes market conditions and past evidence. It is not advice to buy or sell.")
    st.caption("The other tabs show the full technical details behind these numbers.")

with overview:
    close = df["close"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Last close (USDT)", f"{close.iloc[-1]:,.2f}")
    col2.metric("1-day change", f"{close.iloc[-1] / close.iloc[-2] - 1:+.2%}")
    col3.metric("30-day change", f"{close.iloc[-1] / close.iloc[-31] - 1:+.2%}")

    price_fig = px.line(in_range(df), y="close", log_y=True, title=f"{symbol} daily close (log scale)")
    st.plotly_chart(price_fig, width="stretch")

    volume_fig = px.bar(in_range(df), y="volume", title=f"{symbol} daily volume (base asset units)")
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
        in_range(df).dropna(subset=["trend"]), y="close", color="trend", log_y=True,
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
        col1.metric(f"Typical 5-day {swing}", f"±{forecast.iloc[-1] * np.sqrt(5):.1%}",
                    help=f"{swing_long.capitalize()}. Method: {forecast_method}.")
        col2.metric("Past episodes", e["past_episodes"])
        if e["past_episodes"] > 1:
            col3.metric(
                "Past avg 5-day return",
                f"{e['past_mean_return']:+.2%}",
                help=f"Standard error ±{e['past_mean_se']:.2%}",
            )
            col4.metric("Avg day in same trend regime", f"{e['base_same_trend_mean']:+.2%}")
            gap, gap_se = evidence_gap(e)
            st.caption(
                f"Difference from an average {e['trend']}-trend day: {gap:+.2%} "
                f"({gap / gap_se:+.1f} standard errors). "
                "Below about 2 standard errors, the difference could be chance."
            )

        flags = setup_flag(df, setup)
        starts = flags & ~flags.shift(fill_value=False)
        recent = starts[starts].index[-5:]
        st.write("Most recent episode starts: " + ", ".join(str(d.date()) for d in recent))

with volatility:
    HORIZON = 5
    latest = forecast.iloc[-1]

    st.subheader(f"Volatility forecast for the next {HORIZON} days ({last_date.date()})")
    st.write(f"Method: **{forecast_method}**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Daily volatility", f"{latest:.2%}")
    col2.metric(f"Typical {HORIZON}-day {swing}", f"±{latest * np.sqrt(HORIZON):.1%}",
                help=swing_long.capitalize() + ".")
    col3.metric("Annualised", f"{latest * np.sqrt(365):.0%}")
    if is_har:
        st.caption(
            "HAR forecast from 15-minute realized volatility. On 2024+ held-out data it improved forecast "
            "error on realized volatility by 21% (BTC) and 11% (SOL) versus the previous forecast; it was "
            "not better at forecasting where the price closes."
        )
    else:
        note = ("No intraday data for this asset; using the 20-day volatility forecast."
                if symbol not in HAR_SYMBOLS else
                "15-minute data is missing or out of date; using the 20-day volatility forecast. "
                "Click **Refresh data from Binance** to update it.")
        st.caption(
            "Forecast = 20-day volatility × a scale factor learned only from past, completed outcomes. "
            "In Phase 4 (BTC), this scaled forecast beat raw 20-day volatility by 8–11% in every "
            "validation and test year; a Random Forest performed comparably and was not adopted. " + note
        )

    realized_rv = load_realized_ahead(symbol) if symbol in HAR_SYMBOLS else None
    if realized_rv is not None:
        realized, realized_name = realized_rv, f"realized next {HORIZON} days (15-minute data)"
    else:
        realized = rms_ahead(df["log_return"], HORIZON)
        realized_name = f"realized next {HORIZON} days"
    comparison = in_range(pd.DataFrame({
        "forecast": forecast,
        realized_name: realized,
    })) * np.sqrt(365)
    vol_fig = px.line(comparison, title="Forecast vs realized volatility (annualised)")
    vol_fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(vol_fig, width="stretch")
    st.caption(
        f"Realized volatility is only known {HORIZON} days later, so the most recent "
        "days show the forecast alone."
    )

with backtest:
    st.caption(
        "Continuation setup, long only: enter at the next day's open after a signal, hold 5 or 10 days, "
        f"one position at a time, all-in, {2 * COST_PER_SIDE:.2%} round-trip costs. Past performance does not "
        "guarantee future results."
    )
    if symbol in REPLICATION_NOTES:
        st.warning(REPLICATION_NOTES[symbol])
        
    period = st.radio("Period", ["Full history", "Held-out test (2024 onward)"], horizontal=True)
    start = "2024-01-01" if period.startswith("Held-out") else None
    summary_table, curves = run_strategies(symbol, start)

    percent_columns = ["total_return", "annual_return", "annual_volatility", "max_drawdown",
                       "time_in_market", "win_rate", "avg_trade"]
    st.dataframe(
        summary_table.style.format(
            {c: "{:.1%}" for c in percent_columns} | {"trades": "{:.0f}", "return_per_risk": "{:.2f}"},
            na_rep="—")
    )
    st.caption("return_per_risk = annual_return divided by annual_volatility.")

    equity_fig = px.line(in_range(curves), log_y=True, title="Account value (1.0 at the start of the backtest, log scale)")
    st.plotly_chart(equity_fig, width="stretch")

    if symbol == "BTC/USDT":
        st.caption(
            "Phase 6 findings (BTC): the continuation strategy stayed profitable after costs on held-out "
            "data, with far smaller drawdowns and a better return per unit of risk than buy-and-hold, but "
            "a lower total return. Its advantage over random trades taken during uptrends is suggestive, "
            "not established; trade counts are small."
        )
