import math

import numpy as np
import pandas as pd
import pytest

from cryptosense.assistant import build_fact_sheet
from cryptosense.context import derive_insights
from cryptosense.facts import US_OPEN_TERCILES, build_facts, classify_us_open

NOW = pd.Timestamp("2026-10-23 12:00", tz="UTC")


def make_daily(n=60, last="2026-10-22"):
    index = pd.date_range(end=last, periods=n, freq="D", tz="UTC", name="date")
    close = pd.Series(100.0 * 1.01 ** np.arange(n), index=index)
    return pd.DataFrame({
        "close": close, "high": close * 1.02, "low": close * 0.98,
        "trend": "up", "vol_regime": "high",
    }, index=index)


def test_basic_facts():
    facts = build_facts(make_daily(), "SOL/USDT", 0.03, "HAR (realized volatility)", now=NOW)
    assert facts["data_date"] == "2026-10-22"
    assert facts["data_age_days"] == 1
    assert facts["price"] == pytest.approx(100 * 1.01 ** 59)
    assert facts["change_1d"] == pytest.approx(0.01)
    assert facts["change_7d"] == pytest.approx(1.01 ** 7 - 1)
    assert facts["trend"] == "up" and facts["vol_regime"] == "high"
    assert facts["next_weekday"] == "Friday"
    assert facts["range_low"] == pytest.approx(facts["price"] * math.exp(-0.03))


def test_high_low_use_last_30_days():
    daily = make_daily()
    facts = build_facts(daily, "SOL/USDT", 0.03, "HAR", now=NOW)
    assert facts["high_30d"] == pytest.approx(daily["high"].iloc[-30:].max())
    assert facts["low_30d"] == pytest.approx(daily["low"].iloc[-30:].min())


def test_short_history_gives_none_not_errors():
    facts = build_facts(make_daily(n=5), "SOL/USDT", 0.03, "HAR", now=NOW)
    assert facts["change_7d"] is None and facts["change_30d"] is None


def test_missing_labels_and_nan_become_none():
    daily = make_daily()
    daily.loc[daily.index[-1], ["trend", "vol_regime"]] = np.nan
    facts = build_facts(daily, "SOL/USDT", float("nan"), "HAR", now=NOW)
    assert facts["trend"] is None and facts["vol_regime"] is None
    assert facts["swing_1d"] is None and facts["range_low"] is None
    for value in facts.values():
        assert not (isinstance(value, float) and math.isnan(value))


@pytest.mark.parametrize("rv, expected", [(0.012285, "low"), (0.0123, "middle"),
                                          (0.021055, "middle"), (0.0211, "high"), (None, None)])
def test_us_open_terciles_sol(rv, expected):
    assert classify_us_open("SOL/USDT", rv) == expected


def test_us_open_unavailable_for_eth():
    assert classify_us_open("ETH/USDT", 0.05) is None
    assert set(US_OPEN_TERCILES) == {"BTC/USDT", "SOL/USDT"}


def test_track_record_matches_method():
    har = build_facts(make_daily(), "SOL/USDT", 0.03, "HAR (realized volatility)", now=NOW)
    fallback = build_facts(make_daily(), "ETH/USDT", 0.03, "20-day volatility (scaled)", now=NOW)
    assert "11%" in har["forecast_track_record"]
    assert "no intraday data" in fallback["forecast_track_record"]


def test_patterns_are_copied():
    patterns = [{"name": "continuation", "description": "a steady rise", "evidence": "Weak evidence"}]
    facts = build_facts(make_daily(), "SOL/USDT", 0.03, "HAR", now=NOW, patterns=patterns)
    assert facts["active_patterns"] == patterns and facts["active_patterns"] is not patterns


def test_works_end_to_end_with_insights_and_fact_sheet():
    facts = build_facts(make_daily(last="2026-10-23"), "SOL/USDT", 0.012, "HAR", now=NOW, rv_open=0.03)
    insights = derive_insights(facts)
    sheet = build_fact_sheet(facts, insights)
    assert facts["next_weekday"] == "Saturday"
    assert "Tomorrow is Saturday" in sheet
    assert "Active US market open today" in sheet
    assert "data_date: 2026-10-23" in sheet


def test_empty_data_rejected():
    with pytest.raises(ValueError):
        build_facts(make_daily().iloc[:0], "SOL/USDT", 0.03, "HAR", now=NOW)


def test_input_not_modified():
    daily = make_daily()
    before = daily.copy()
    build_facts(daily, "SOL/USDT", 0.03, "HAR", now=NOW)
    pd.testing.assert_frame_equal(daily, before)
