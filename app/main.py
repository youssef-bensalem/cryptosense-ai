"""CryptoSense AI dashboard. Run from the project root with: streamlit run app/main.py"""

from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from cryptosense.data import load_processed
from cryptosense.features import add_features
from cryptosense.regimes import add_regimes

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


@st.cache_data
def load_symbol(symbol):
    """Processed data with features and regimes, cached between reruns."""
    return add_regimes(add_features(load_processed(symbol, DATA_DIR)))


st.set_page_config(page_title="CryptoSense AI", layout="wide")
st.title("CryptoSense AI")
st.caption("Research and decision support only. Not financial advice; no trades are placed.")

symbol = st.sidebar.selectbox("Asset", WATCHLIST)
df = load_symbol(symbol)

last_date = df.index[-1]
age_days = (pd.Timestamp.now(tz="UTC").normalize() - last_date).days
st.sidebar.write(f"Last complete candle: **{last_date.date()}**")
st.sidebar.write(f"Data age: {age_days} day(s)")

overview, conditions, setups, volatility, backtest = st.tabs(
    ["Market overview", "Market conditions", "Setup research", "Volatility", "Backtesting"]
)

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

for tab in (setups, volatility, backtest):
    with tab:
        st.info("Coming in the next milestones.")
