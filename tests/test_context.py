import math

import pytest

from cryptosense.context import ROUND_TRIP_COST, Insight, derive_insights, describe_swing

BASE = {"asset": "SOL/USDT", "swing_1d": 0.03, "vol_regime": "low",
        "us_open_activity": "middle", "next_weekday": "Tuesday", "data_age_days": 0}


def titles(facts):
    return [i.title for i in derive_insights(facts)]


def test_describe_swing_range():
    r = describe_swing(100.0, 0.05)
    assert r["low"] == pytest.approx(100 * math.exp(-0.05))
    assert r["high"] == pytest.approx(100 * math.exp(0.05))


@pytest.mark.parametrize("price, vol", [(0, 0.02), (100, 0), (-1, 0.02), (None, 0.02)])
def test_describe_swing_rejects_invalid(price, vol):
    with pytest.raises(ValueError):
        describe_swing(price, vol)


def test_always_states_no_direction_forecast():
    assert titles(BASE)[-1] == "No direction forecast"


def test_never_predicts_direction():
    for facts in [BASE, {**BASE, "vol_regime": "high", "us_open_activity": "high", "next_weekday": "Saturday"}]:
        for insight in derive_insights(facts):
            text = insight.text.lower()
            for phrase in ["will rise", "will fall", "buy now", "sell now", "guaranteed"]:
                assert phrase not in text


def test_cost_share_thresholds():
    assert "Small expected range relative to costs" in titles({**BASE, "swing_1d": ROUND_TRIP_COST / 0.20})
    assert "Small expected range relative to costs" not in titles({**BASE, "swing_1d": ROUND_TRIP_COST / 0.19})
    assert "Costs are small relative to the typical range" in titles({**BASE, "swing_1d": ROUND_TRIP_COST / 0.10})


def test_high_volatility_insight():
    assert "Turbulent conditions tend to persist" in titles({**BASE, "vol_regime": "high"})


def test_weekend_and_scope_note():
    sol = derive_insights({**BASE, "next_weekday": "Saturday"})
    btc = derive_insights({**BASE, "asset": "BTC/USDT", "next_weekday": "Saturday"})
    sat_sol = [i for i in sol if i.title == "Tomorrow is Saturday"][0]
    sat_btc = [i for i in btc if i.title == "Tomorrow is Saturday"][0]
    assert "not been tested on this asset" in sat_sol.text
    assert "not been tested" not in sat_btc.text


def test_stale_data_warning_first():
    insights = derive_insights({**BASE, "data_age_days": 3})
    assert insights[0].kind == "warning"


def test_missing_facts_skip_insights():
    insights = derive_insights({"asset": "SOL/USDT"})
    assert [i.title for i in insights] == ["No direction forecast"]


def test_every_insight_names_its_evidence():
    facts = {**BASE, "vol_regime": "high", "us_open_activity": "high",
             "next_weekday": "Sunday", "data_age_days": 2, "swing_1d": 0.01}
    for insight in derive_insights(facts):
        assert isinstance(insight, Insight) and insight.evidence
