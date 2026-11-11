"""Shared building blocks: one cost constant, one rule engine, one outcome formula, one evidence gap."""
import math

import numpy as np
import pandas as pd
import pytest

from cryptosense import backtest, context, facts, intraday, one_day, setups, volatility
from cryptosense.backtest import COST_PER_SIDE, run_backtest, summarize_backtest
from cryptosense.explain import evidence_gap, evidence_strength, setup_evidence
from cryptosense.features import add_features
from cryptosense.regimes import add_regimes
from cryptosense.setups import OPS, SETUPS, conditions_hold, explain_setup, setup_flag
from cryptosense.volatility import rms_ahead, scaled_vol_forecast
from test_features import make_random_df


# --- costs -----------------------------------------------------------------

def test_cost_assumption_has_one_source():
    assert COST_PER_SIDE == 0.0015
    assert context.ROUND_TRIP_COST == 2 * COST_PER_SIDE == 0.003
    assert run_backtest.__defaults__ == (COST_PER_SIDE,)


# --- backtest summary --------------------------------------------------------

def make_equity(values):
    index = pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=index, dtype=float)


def no_trades():
    return pd.DataFrame(columns=["entry_date", "exit_date", "entry_price", "exit_price", "gross_return", "net_return"])


def test_annual_return_and_return_per_risk_known_values():
    # 1.0 -> 1.21 over 730 days is 10% a year.
    values = np.ones(731)
    values[365:] = 1.1
    values[-1] = 1.21
    summary = summarize_backtest(no_trades(), make_equity(values))
    assert summary["annual_return"] == pytest.approx(0.10)
    daily = pd.Series(values).pct_change().dropna()
    assert summary["annual_volatility"] == pytest.approx(daily.std() * math.sqrt(365))
    assert summary["return_per_risk"] == pytest.approx(0.10 / summary["annual_volatility"])


def test_return_per_risk_is_none_without_risk():
    summary = summarize_backtest(no_trades(), make_equity([1.0, 1.0, 1.0]))
    assert summary["annual_return"] == pytest.approx(0.0)
    assert summary["return_per_risk"] is None


# --- rule engine -------------------------------------------------------------

def test_every_module_uses_the_shared_rule_engine():
    assert set(OPS) == {">", "<", "=="}
    assert intraday.conditions_hold is conditions_hold
    assert one_day.conditions_hold is conditions_hold
    for module in (intraday, one_day):
        assert not hasattr(module, "INTRADAY_OPS") and not hasattr(module, "ONE_DAY_OPS")


def test_conditions_hold_known_values_and_missing_values():
    frame = pd.DataFrame({"a": [1.0, 2.0, np.nan, 3.0], "label": ["up", "down", "up", None]})
    assert conditions_hold(frame, [("a", ">", 1.5)]).tolist() == [False, True, False, True]
    assert conditions_hold(frame, [("a", "<", 2.5), ("label", "==", "up")]).tolist() == [True, False, False, False]
    assert conditions_hold(frame, []).tolist() == [True] * 4
    assert conditions_hold(frame, [("a", ">", 0.0)]).dtype == bool


def test_setup_flag_is_the_shared_engine_on_the_setup_rules():
    df = add_regimes(add_features(make_random_df(800)))
    for setup in SETUPS:
        assert setup_flag(df, setup).equals(conditions_hold(df, SETUPS[setup]))


# --- outcome formula ---------------------------------------------------------

def test_rms_ahead_known_values():
    values = pd.Series([0.0, 3.0, 4.0, 0.0, 0.0])
    result = rms_ahead(values, 2)
    assert result.iloc[0] == pytest.approx(math.sqrt((9 + 16) / 2))
    assert result.iloc[1] == pytest.approx(math.sqrt(16 / 2))
    assert result.iloc[2] == 0.0
    assert result.iloc[3:].isna().all()


def test_rms_ahead_uses_exactly_the_next_horizon_rows():
    rng = np.random.default_rng(1)
    values = pd.Series(rng.normal(0, 0.02, 300))
    horizon = 5
    full = rms_ahead(values, horizon)
    for t in (10, 150, 290):
        # Truncation: nothing after t + horizon matters, and row t itself does not matter.
        truncated = rms_ahead(values.iloc[:t + horizon + 1], horizon)
        assert truncated.iloc[t] == full.iloc[t]
        changed = values.copy()
        changed.iloc[t] = 0.5
        # approx: pandas' rolling sum keeps a rounding residue of the value that left the window
        assert rms_ahead(changed, horizon).iloc[t] == pytest.approx(full.iloc[t], rel=1e-9)
        assert rms_ahead(changed, horizon).iloc[t - 1] != pytest.approx(full.iloc[t - 1], rel=1e-9)


def test_explain_setup_reports_the_scaled_forecast():
    df = add_regimes(add_features(make_random_df(800)))
    for position in (450, 799):
        date = df.index[position]
        explanation = explain_setup(df, date, "continuation")
        expected = scaled_vol_forecast(df.loc[:date], 5).iloc[-1]
        assert explanation["daily_vol_forecast"] == round(float(expected), 4)
        assert explanation["typical_move"] == round(float(expected * np.sqrt(5)), 4)


def test_modules_no_longer_carry_their_own_copy_of_the_outcome_formula():
    assert setups.scaled_vol_forecast is volatility.scaled_vol_forecast


# --- US-open threshold -------------------------------------------------------

def test_us_open_high_tercile_is_the_preregistered_threshold():
    for symbol, threshold in one_day.RV_OPEN_THRESHOLDS.items():
        assert facts.US_OPEN_TERCILES[symbol][1] == threshold
        assert facts.classify_us_open(symbol, threshold) == "middle"       # equal: not above
        assert facts.classify_us_open(symbol, threshold * 1.0001) == "high"


# --- evidence gap ------------------------------------------------------------

def test_evidence_gap_and_label():
    explanation = {"past_mean_return": 0.05, "base_same_trend_mean": 0.01, "past_mean_se": 0.01,
                   "past_episodes": 40}
    gap, se = evidence_gap(explanation)
    assert gap == pytest.approx(0.04) and se == 0.01
    assert setup_evidence(explanation) == evidence_strength(gap, se, 40) == "Stronger evidence"


@pytest.mark.parametrize("missing", ["past_mean_return", "base_same_trend_mean", "past_mean_se"])
def test_evidence_gap_is_nan_when_a_statistic_is_missing(missing):
    explanation = {"past_mean_return": 0.05, "base_same_trend_mean": 0.01, "past_mean_se": 0.01,
                   "past_episodes": 3, missing: None}
    gap, se = evidence_gap(explanation)
    assert math.isnan(gap) and math.isnan(se)
    assert setup_evidence(explanation) == "Too little data"


def test_setup_evidence_matches_explain_setup_output():
    df = add_regimes(add_features(make_random_df(800)))
    for setup in SETUPS:
        explanation = explain_setup(df, df.index[-1], setup)
        assert setup_evidence(explanation) in {"Too little data", "Did worse than average", "No evidence",
                                               "Weak evidence", "Some evidence", "Stronger evidence"}
