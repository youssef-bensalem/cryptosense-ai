"""The service layer: every payload comes from the tested modules, is JSON-safe and uses only past data."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cryptosense import service
from cryptosense.backtest import COST_PER_SIDE, run_backtest, summarize_backtest
from cryptosense.context import derive_insights
from cryptosense.data import DAY_MS, candles_to_dataframe, save_dataset
from cryptosense.explain import position_size, stop_level, suggested_stop, typical_move_range
from cryptosense.features import add_features
from cryptosense.regimes import add_regimes
from cryptosense.setups import setup_flag
from cryptosense.volatility import HAR_LABEL, scaled_vol_forecast

LAST_DAY = pd.Timestamp("2026-10-22", tz="UTC")
NOW = pd.Timestamp("2026-10-23 12:00", tz="UTC")
M15_MS = 15 * 60 * 1000


def make_daily_candles(days, seed, last_day=LAST_DAY):
    """Daily candles ending on ``last_day``: a random walk with valid highs, lows and volume."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, days)))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.random(days) * 0.01)
    low = np.minimum(open_, close) * (1 - rng.random(days) * 0.01)
    volume = rng.random(days) * 1000 + 100
    start = int(last_day.timestamp() * 1000) - (days - 1) * DAY_MS
    return [[start + i * DAY_MS, float(open_[i]), float(high[i]), float(low[i]), float(close[i]), float(volume[i])]
            for i in range(days)]


def make_m15_candles(days, seed, last_day=LAST_DAY):
    """Complete 15-minute candles for ``days`` UTC days ending with ``last_day``."""
    rng = np.random.default_rng(seed)
    n = days * 96
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    start = int(last_day.timestamp() * 1000) - (days - 1) * DAY_MS
    return [[start + i * M15_MS, float(open_[i]), float(max(open_[i], close[i])), float(min(open_[i], close[i])),
             float(close[i]), 10.0] for i in range(n)]


def write_data(data_dir, days=800, m15_days=0, symbols=tuple(service.WATCHLIST), last_day=LAST_DAY):
    """Save synthetic daily data for ``symbols`` (and 15-minute data for SOL if ``m15_days``)."""
    for seed, symbol in enumerate(symbols):
        candles = make_daily_candles(days, seed, last_day)
        save_dataset(candles, candles_to_dataframe(candles), symbol, data_dir)
    if m15_days:
        candles = make_m15_candles(m15_days, 99, last_day)
        save_dataset(candles, candles_to_dataframe(candles), "SOL/USDT", data_dir, timeframe="15m")


@pytest.fixture(autouse=True)
def fresh_cache():
    service.clear_cache()
    yield
    service.clear_cache()


@pytest.fixture
def data_dir(tmp_path):
    write_data(tmp_path)
    return tmp_path


def all_payloads(data_dir, symbol):
    return {
        "today": service.today(data_dir, symbol, NOW),
        "week": service.week(data_dir, symbol),
        "prices": service.prices(data_dir, symbol, "all"),
        "conditions": service.conditions(data_dir, symbol),
        "patterns": service.patterns(data_dir, symbol),
        "volatility": service.volatility(data_dir, symbol, "all"),
        "strategies": service.strategies(data_dir, symbol, "full", "all"),
    }


class FakeClient:
    def __init__(self, reply="A plain explanation.", error=None):
        self.requests, self.reply, self.error = [], reply, error
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


# --- helpers -----------------------------------------------------------------

def test_clean_makes_values_json_safe():
    raw = {"a": np.float64(1.5), "b": float("nan"), "c": np.int64(3), "d": [np.inf, None, pd.NaT],
           "e": pd.Timestamp("2026-01-02", tz="UTC"), "f": np.bool_(True), "g": ("x", 2), 5: "five"}
    cleaned = service.clean(raw)
    assert cleaned == {"a": 1.5, "b": None, "c": 3, "d": [None, None, None],
                       "e": "2026-01-02T00:00:00+00:00", "f": True, "g": ["x", 2], "5": "five"}
    assert type(cleaned["a"]) is float and type(cleaned["c"]) is int and cleaned["f"] is True
    json.dumps(cleaned, allow_nan=False)


def test_slug_to_symbol():
    assert service.slug_to_symbol("sol_usdt") == "SOL/USDT"
    for bad in ["doge_usdt", "SOL/USDT", "", "../etc"]:
        with pytest.raises(service.UnknownAsset):
            service.slug_to_symbol(bad)


def test_missing_data_raises_no_data(tmp_path):
    assert service.has_data(tmp_path, "SOL/USDT") is False
    with pytest.raises(service.NoData):
        service.today(tmp_path, "SOL/USDT", NOW)
    assert [a["has_data"] for a in service.list_assets(tmp_path, NOW)] == [False, False, False]


def test_unknown_range_and_period_rejected(data_dir):
    with pytest.raises(ValueError):
        service.prices(data_dir, "SOL/USDT", "5y")
    with pytest.raises(ValueError):
        service.strategies(data_dir, "SOL/USDT", "2023")


# --- every payload -----------------------------------------------------------

@pytest.mark.parametrize("symbol", service.WATCHLIST)
def test_every_payload_is_strict_json(data_dir, symbol):
    for payload in all_payloads(data_dir, symbol).values():
        json.dumps(payload, allow_nan=False)
    json.dumps(service.list_assets(data_dir, NOW), allow_nan=False)


def test_payloads_survive_too_little_history(tmp_path):
    write_data(tmp_path, days=40)
    payloads = all_payloads(tmp_path, "ETH/USDT")
    for payload in payloads.values():
        json.dumps(payload, allow_nan=False)
    assert payloads["today"]["trend"] is None
    assert payloads["today"]["trend_words"] == "not enough history yet"
    assert payloads["week"]["above_sma_100"] is None
    assert payloads["strategies"]["summary"] == [] and payloads["strategies"]["points"] == []


def test_list_assets(data_dir):
    assets = service.list_assets(data_dir, NOW)
    assert [a["symbol"] for a in assets] == service.WATCHLIST
    assert [a["slug"] for a in assets] == ["sol_usdt", "btc_usdt", "eth_usdt"]
    assert all(a["has_data"] and a["data_date"] == "2026-10-22" and a["data_age_days"] == 1 for a in assets)


# --- today -------------------------------------------------------------------

def test_today_matches_facts_and_insights(data_dir):
    facts, insights = service.facts_and_insights(data_dir, "BTC/USDT", NOW)
    payload = service.today(data_dir, "BTC/USDT", NOW)
    assert payload["price"] == facts["price"]
    assert payload["swing_1d"] == facts["swing_1d"]
    assert (payload["range_low"], payload["range_high"]) == (facts["range_low"], facts["range_high"])
    assert payload["tomorrow"] == "2026-10-23" and payload["next_weekday"] == "Friday"
    assert {i["title"] for i in payload["insights"]} == {i.title for i in derive_insights(facts)}
    assert payload["intraday"] is None and payload["intraday_supported"] is True
    assert service.today(data_dir, "ETH/USDT", NOW)["intraday_supported"] is False


def test_today_uses_the_scaled_forecast_without_intraday_data(data_dir):
    df = add_regimes(add_features(candles_to_dataframe(make_daily_candles(800, 1))))
    payload = service.today(data_dir, "BTC/USDT", NOW)
    assert payload["swing_1d"] == pytest.approx(scaled_vol_forecast(df, 1).iloc[-1])
    assert payload["swing_word"] == "move"


def test_today_puts_warnings_first(data_dir):
    stale = pd.Timestamp("2026-10-30 12:00", tz="UTC")
    kinds = [i["kind"] for i in service.today(data_dir, "SOL/USDT", stale)["insights"]]
    assert kinds[0] == "warning" and "warning" not in kinds[1:]


def test_today_with_intraday_data_uses_har_and_shows_three_days(tmp_path):
    write_data(tmp_path, days=800, m15_days=420)
    payload = service.today(tmp_path, "SOL/USDT", NOW)
    assert payload["method"] == HAR_LABEL and payload["swing_word"] == "swing"
    times = [pd.Timestamp(point["t"]) for point in payload["intraday"]]
    assert times[-1] == LAST_DAY + pd.Timedelta(hours=23, minutes=45)
    assert times[0] == times[-1] - pd.Timedelta(days=service.INTRADAY_DAYS)
    assert len(times) == service.INTRADAY_DAYS * 96 + 1
    json.dumps(service.volatility(tmp_path, "SOL/USDT", "1y"), allow_nan=False)
    assert "15-minute" in service.volatility(tmp_path, "SOL/USDT")["realized_name"]


# --- week, prices, conditions, patterns, volatility ----------------------------

def test_week_matches_typical_move_range(data_dir):
    df = service.load_daily(data_dir, "BTC/USDT")
    forecast, _ = service.load_forecast(data_dir, "BTC/USDT", 5)
    move, low, high = typical_move_range(df["close"].iloc[-1], forecast.iloc[-1], horizon=5)
    payload = service.week(data_dir, "BTC/USDT")
    assert (payload["move"], payload["range_low"], payload["range_high"]) == (move, low, high)
    assert payload["change_20d"] == pytest.approx(df["close"].iloc[-1] / df["close"].iloc[-21] - 1)
    assert [p["name"] for p in payload["patterns"]] == ["pullback", "continuation"]
    for pattern in payload["patterns"]:
        assert (pattern["evidence"] is not None) == pattern["present"]
    assert payload["replication_note"] is None
    assert service.week(data_dir, "SOL/USDT")["replication_note"] == service.REPLICATION_NOTES["SOL/USDT"]


def test_prices_ranges(data_dir):
    everything = service.prices(data_dir, "SOL/USDT", "all")
    month = service.prices(data_dir, "SOL/USDT", "1m")
    assert len(everything["points"]) == 800
    assert 28 <= len(month["points"]) <= 32
    assert month["points"] == everything["points"][-len(month["points"]):]
    assert everything["points"][-1]["t"] == "2026-10-22"
    assert everything["points"][-1]["close"] == everything["last_close"]
    assert everything["points"][0]["trend"] is None and everything["points"][-1]["trend"] in ("up", "down", "range")


def test_conditions_match_the_feature_columns(data_dir):
    latest = service.load_daily(data_dir, "ETH/USDT").iloc[-1]
    payload = service.conditions(data_dir, "ETH/USDT")
    assert payload["momentum_20d"] == pytest.approx(math.exp(latest["mom_20"]) - 1)
    assert payload["distance_sma_100"] == latest["dist_sma_100"]
    assert payload["volatility_20d_annual"] == pytest.approx(latest["vol_20"] * math.sqrt(365))
    assert sum(payload["trend_shares"].values()) == pytest.approx(1.0)


def test_patterns_report_the_setup_rules(data_dir):
    payload = service.patterns(data_dir, "BTC/USDT")
    by_name = {s["name"]: s for s in payload["setups"]}
    assert [c["feature"] for c in by_name["continuation"]["conditions"]] == ["mom_20", "er_20", "rel_volume_20"]
    df = service.load_daily(data_dir, "BTC/USDT")
    for name, setup in by_name.items():
        assert setup["flagged"] == bool(setup_flag(df, name).iloc[-1])
        assert len(setup["recent_starts"]) <= 5
        if setup["gap"] is not None:
            assert setup["gap"] == pytest.approx(setup["past_mean_return"] - setup["base_same_trend_mean"])


def test_volatility_series_is_annualised_forecast(data_dir):
    forecast, method = service.load_forecast(data_dir, "ETH/USDT", 5)
    payload = service.volatility(data_dir, "ETH/USDT", "all")
    assert payload["daily"] == forecast.iloc[-1]
    assert payload["points"][-1]["forecast"] == pytest.approx(forecast.iloc[-1] * math.sqrt(365))
    assert payload["points"][-1]["realized"] is None       # the outcome of the last day is not known yet
    assert payload["points"][-10]["realized"] is not None
    assert service.NO_INTRADAY_NOTE in payload["caption"]


# --- strategies and risk -----------------------------------------------------

def test_strategies_match_a_direct_backtest(data_dir):
    data = service.load_daily(data_dir, "BTC/USDT").dropna(subset=["dist_sma_100", "rel_volume_20", "er_20"])
    trades, equity = run_backtest(data, setup_flag(data, "continuation"), 5, COST_PER_SIDE)
    expected = summarize_backtest(trades, equity)
    payload = service.strategies(data_dir, "BTC/USDT", "full", "all")
    row = next(r for r in payload["summary"] if r["name"] == "continuation 5d")
    for key in ("total_return", "annual_return", "max_drawdown", "trades", "win_rate"):
        assert row[key] == pytest.approx(expected[key])
    assert payload["points"][-1]["continuation 5d"] == pytest.approx(equity.iloc[-1])
    assert payload["strategies"] == ["continuation 5d", "continuation 10d", "buy and hold"]
    assert next(r for r in payload["summary"] if r["name"] == "buy and hold")["win_rate"] is None
    assert payload["findings"] == service.BTC_STRATEGY_FINDINGS
    assert service.strategies(data_dir, "SOL/USDT")["findings"] is None


def test_held_out_period_starts_in_2024(data_dir):
    points = service.strategies(data_dir, "BTC/USDT", "heldout", "all")["points"]
    assert points[0]["t"] >= service.HELD_OUT_START
    assert points[0]["buy and hold"] == pytest.approx(1.0)


def test_risk_matches_the_explain_functions():
    result = service.risk(price=180.0, typical_move=0.03, account=1000.0, risk_pct=1.0)
    stop_price, stop_fraction = stop_level(180.0, suggested_stop(0.03))
    assert result["stop_price"] == stop_price and result["stop_fraction"] == stop_fraction
    assert result["invest"] == position_size(1000.0, 0.01, stop_fraction)
    assert result["loss_at_stop"] == 10.0 and result["exceeds_account"] is False
    assert service.risk(180.0, 0.001, 1000.0, 5.0)["exceeds_account"] is True


@pytest.mark.parametrize("arguments", [(0, 0.03, 1000, 1), (180, -0.1, 1000, 1), (180, 0.03, 0, 1),
                                       (180, 0.03, 1000, 0), (180, 0.03, 1000, 101), (float("nan"), 0.03, 1000, 1),
                                       (180, float("inf"), 1000, 1)])
def test_risk_rejects_bad_inputs(arguments):
    with pytest.raises(ValueError):
        service.risk(*arguments)


# --- assistant ---------------------------------------------------------------

def test_trade_and_direction_questions_get_the_fixed_answer(data_dir):
    client = FakeClient(reply="You should buy.")
    for question in ["Should I buy SOL now?", "Will it go up tomorrow?"]:
        result = service.answer(data_dir, "SOL/USDT", question, client=client, now=NOW)
        assert result["source"] == "fixed"
        assert result["answer"].endswith("This is information, not financial advice.")
    assert client.requests == []
    assert service.answer(data_dir, "SOL/USDT", "Should I sell?", client=None, now=NOW)["source"] == "fixed"


def test_ordinary_question_goes_to_the_model_with_history(data_dir):
    client = FakeClient()
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    result = service.answer(data_dir, "SOL/USDT", "What is a stop?", history, client, NOW)
    assert result == {"answer": "A plain explanation.", "source": "model"}
    assert client.requests[0]["messages"][:2] == history
    assert "FACTS" in client.requests[0]["system"]


def test_directional_model_answer_is_withheld(data_dir):
    result = service.answer(data_dir, "SOL/USDT", "What is a stop?", client=FakeClient("SOL will rise tomorrow."),
                            now=NOW)
    assert result["source"] == "withheld" and "will rise" not in result["answer"]


def test_no_model_and_failing_model_are_reported_as_unavailable(data_dir):
    assert service.answer(data_dir, "SOL/USDT", "What is a stop?", client=None, now=NOW) == {
        "answer": service.MODEL_UNAVAILABLE, "source": "unavailable"}
    failing = service.answer(data_dir, "SOL/USDT", "What is a stop?", client=FakeClient(error=RuntimeError("x")),
                             now=NOW)
    assert failing["source"] == "unavailable" and "RuntimeError" in failing["answer"]


def test_empty_question_rejected(data_dir):
    with pytest.raises(ValueError):
        service.answer(data_dir, "SOL/USDT", "   ", client=None, now=NOW)


# --- caching -----------------------------------------------------------------

def test_results_are_reused_until_the_file_changes(tmp_path):
    write_data(tmp_path, days=300)
    first = service.load_daily(tmp_path, "SOL/USDT")
    assert service.load_daily(tmp_path, "SOL/USDT") is first
    assert service.prices(tmp_path, "SOL/USDT", "all")["points"][-1]["t"] == "2026-10-22"

    write_data(tmp_path, days=301, last_day=LAST_DAY + pd.Timedelta(days=1))
    second = service.load_daily(tmp_path, "SOL/USDT")
    assert second is not first and len(second) == 301
    assert service.prices(tmp_path, "SOL/USDT", "all")["points"][-1]["t"] == "2026-10-23"
    assert service.today(tmp_path, "SOL/USDT", NOW)["data_date"] == "2026-10-23"


# --- truncation leakage --------------------------------------------------------

@pytest.mark.parametrize("cut", [500, 700])
def test_payloads_for_a_day_use_only_data_up_to_that_day(tmp_path, cut):
    """Data that ends on day D must give the same numbers for D as the full data does."""
    full_dir, cut_dir = tmp_path / "full", tmp_path / "cut"
    candles = make_daily_candles(800, 7)
    save_dataset(candles, candles_to_dataframe(candles), "BTC/USDT", full_dir)
    save_dataset(candles[:cut], candles_to_dataframe(candles[:cut]), "BTC/USDT", cut_dir)

    full = service.load_daily(full_dir, "BTC/USDT")
    day = full.index[cut - 1]
    row = full.loc[day]

    conditions = service.conditions(cut_dir, "BTC/USDT")
    assert conditions["data_date"] == str(day.date())
    assert conditions["distance_sma_100"] == row["dist_sma_100"]
    assert conditions["trend"] == row["trend"] and conditions["vol_regime"] == row["vol_regime"]

    # Features and labels of past days do not change when later days are added.
    past_points = service.prices(cut_dir, "BTC/USDT", "all")["points"]
    assert past_points == service.prices(full_dir, "BTC/USDT", "all")["points"][:cut]

    # The forecast made on each day is the same with or without later days.
    full_forecast = {p["t"]: p["forecast"] for p in service.volatility(full_dir, "BTC/USDT", "all")["points"]}
    for point in service.volatility(cut_dir, "BTC/USDT", "all")["points"]:
        assert point["forecast"] == pytest.approx(full_forecast[point["t"]], rel=1e-9, nan_ok=True) \
            if point["forecast"] is not None else full_forecast[point["t"]] is None

    week = service.week(cut_dir, "BTC/USDT")
    forecast_then = scaled_vol_forecast(full.loc[:day], 5).iloc[-1]
    assert week["move"] == pytest.approx(forecast_then * math.sqrt(5))

    today = service.today(cut_dir, "BTC/USDT", day + pd.Timedelta(hours=36))
    assert today["price"] == row["close"]
    assert today["swing_1d"] == pytest.approx(scaled_vol_forecast(full.loc[:day], 1).iloc[-1])
