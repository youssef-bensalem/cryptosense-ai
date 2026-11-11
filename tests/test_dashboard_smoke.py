"""Smoke test: the dashboard script runs end to end on synthetic data without errors.

No network: data comes from a temporary folder (CRYPTOSENSE_DATA_DIR), the
refresh button is never clicked, and the local-assistant check is replaced.
"""
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import cryptosense.assistant
from cryptosense.data import DAY_MS, candles_to_dataframe, save_dataset

APP = str(Path(__file__).resolve().parents[1] / "app" / "main.py")
SYMBOLS = ["SOL/USDT", "BTC/USDT", "ETH/USDT"]


def make_candles(days, seed):
    """Daily candles ending yesterday (UTC): a random walk with valid highs, lows and volume."""
    rng = np.random.default_rng(seed)
    today_ms = int(np.datetime64("today", "ms").astype("int64"))
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, days)))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.random(days) * 0.01)
    low = np.minimum(open_, close) * (1 - rng.random(days) * 0.01)
    volume = rng.random(days) * 1000 + 100
    start = today_ms - days * DAY_MS
    return [[start + i * DAY_MS, float(open_[i]), float(high[i]), float(low[i]), float(close[i]), float(volume[i])]
            for i in range(days)]


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    """Return ``run(symbol=None)``, which runs the dashboard on the temporary data folder."""
    monkeypatch.setenv("CRYPTOSENSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cryptosense.assistant, "get_local_client", lambda *args, **kwargs: None)
    st.cache_data.clear()
    st.cache_resource.clear()

    def run(symbol=None):
        app = AppTest.from_file(APP, default_timeout=120).run()
        if symbol is not None:
            app = app.sidebar.selectbox[0].select(symbol).run()
        return app

    yield run
    st.cache_data.clear()
    st.cache_resource.clear()


def write_data(data_dir):
    for seed, symbol in enumerate(SYMBOLS):
        candles = make_candles(800, seed)
        save_dataset(candles, candles_to_dataframe(candles), symbol, data_dir)


def test_dashboard_without_data_asks_for_a_refresh(dashboard):
    app = dashboard()
    assert not app.exception
    assert any("No data for SOL/USDT yet" in info.value for info in app.info)


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_dashboard_runs_for_every_asset(dashboard, tmp_path, symbol):
    write_data(tmp_path)
    app = dashboard(symbol)
    assert not app.exception
    assert [tab.label for tab in app.tabs][:3] == ["Today & tomorrow", "Ask the assistant", "Summary"]
    assert any(symbol in header.value for header in app.subheader)


def test_dashboard_has_one_risk_helper_per_holding_period(dashboard, tmp_path):
    write_data(tmp_path)
    app = dashboard()
    keys = {widget.key for widget in app.number_input}
    assert keys == {"account_1d", "risk_pct_1d", "account_5d", "risk_pct_5d"}
    stops = [metric for metric in app.metric if metric.label == "Suggested stop"]
    assert len(stops) == 2


def test_dashboard_shows_news_collector_line_without_sentiment(dashboard, tmp_path):
    write_data(tmp_path)
    app = dashboard()
    labels = [expander.label for expander in app.sidebar.expander]
    assert any(label.startswith("News collector:") for label in labels)
