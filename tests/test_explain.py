import itertools
import math

import numpy as np
import pandas as pd
import pytest

from cryptosense.explain import (
    EVIDENCE_LABELS,
    describe_trend,
    describe_volatility,
    evidence_strength,
    position_size,
    suggested_stop,
    typical_move_range,
)

MISSING = [None, np.nan, float("nan"), pd.NA]


@pytest.mark.parametrize("label, words", [
    ("up", "rising"),
    ("down", "falling"),
    ("range", "mixed (the signals disagree)"),
])
def test_describe_trend(label, words):
    assert describe_trend(label) == words


@pytest.mark.parametrize("label, words", [
    ("high", "more turbulent than usual (compared with the past year)"),
    ("low", "calmer than usual (compared with the past year)"),
])
def test_describe_volatility(label, words):
    assert describe_volatility(label) == words


@pytest.mark.parametrize("missing", MISSING)
def test_missing_labels_mean_not_enough_history(missing):
    assert describe_trend(missing) == "not enough history yet"
    assert describe_volatility(missing) == "not enough history yet"


def test_unknown_labels_raise():
    with pytest.raises(ValueError):
        describe_trend("high")
    with pytest.raises(ValueError):
        describe_volatility("up")


def test_typical_move_range_known_numbers():
    move, low, high = typical_move_range(100.0, 0.02, horizon=4)
    assert move == pytest.approx(0.04)
    assert low == pytest.approx(100 * math.exp(-0.04))
    assert high == pytest.approx(100 * math.exp(0.04))

    move, low, high = typical_move_range(50_000.0, 0.03)  # default horizon of 5 days
    assert move == pytest.approx(0.03 * math.sqrt(5))
    assert low == pytest.approx(50_000 * math.exp(-0.03 * math.sqrt(5)))
    assert high == pytest.approx(50_000 * math.exp(0.03 * math.sqrt(5)))
    assert low * high == pytest.approx(50_000 ** 2)


@pytest.mark.parametrize("gap, se, n, label", [
    # Number of episodes: 29 is too few whatever the gap, 30 is enough.
    (5.0, 1.0, 29, "Too little data"),
    (-5.0, 1.0, 29, "Too little data"),
    (0.0, 1.0, 29, "Too little data"),
    (5.0, 1.0, 30, "Stronger evidence"),
    # Exactly 1, 2 and 3 standard errors belong to the higher band.
    (0.99, 1.0, 30, "No evidence"),
    (1.0, 1.0, 30, "Weak evidence"),
    (1.99, 1.0, 30, "Weak evidence"),
    (2.0, 1.0, 30, "Some evidence"),
    (2.99, 1.0, 30, "Some evidence"),
    (3.0, 1.0, 30, "Stronger evidence"),
    (0.5, 0.25, 100, "Some evidence"),
    (0.0, 1.0, 30, "No evidence"),
    # Negative gaps: within 1 SE is no evidence; 1 SE or more below zero did worse.
    (-0.99, 1.0, 30, "No evidence"),
    (-1.0, 1.0, 30, "Did worse than average"),
    (-2.5, 1.0, 30, "Did worse than average"),
    (-0.5, 0.25, 100, "Did worse than average"),
])
def test_evidence_strength_boundaries(gap, se, n, label):
    assert evidence_strength(gap, se, n) == label


def test_evidence_strength_returns_only_the_six_labels():
    assert set(EVIDENCE_LABELS) == {"Too little data", "Did worse than average", "No evidence",
                                    "Weak evidence", "Some evidence", "Stronger evidence"}
    gaps = np.concatenate([np.linspace(-5, 5, 401), [-1e9, -1e-12, 0.0, 1e-12, 1e9]])
    ses = [1e-6, 0.01, 0.5, 1.0, 3.0, 1e6]
    ns = [0, 1, 29, 30, 31, 1000]
    seen = {evidence_strength(g, s, n) for g, s, n in itertools.product(gaps, ses, ns)}
    assert seen <= set(EVIDENCE_LABELS)
    assert seen == set(EVIDENCE_LABELS)


@pytest.mark.parametrize("gap, se", [(np.nan, 1.0), (1.0, np.nan), (1.0, 0.0), (1.0, -1.0), (np.inf, 1.0)])
def test_evidence_strength_rejects_invalid_inputs(gap, se):
    with pytest.raises(ValueError):
        evidence_strength(gap, se, 30)


def test_position_size_known_numbers():
    assert position_size(10_000, 0.01, 0.05) == pytest.approx(2_000)
    assert position_size(5_000, 0.02, 0.10) == pytest.approx(1_000)
    # Hitting the stop loses exactly account x risk_fraction.
    size = position_size(12_345, 0.015, 0.0732)
    assert size * 0.0732 == pytest.approx(12_345 * 0.015)


@pytest.mark.parametrize("account, risk, stop", [
    (0, 0.01, 0.05), (-1, 0.01, 0.05),
    (10_000, 0, 0.05), (10_000, -0.01, 0.05),
    (10_000, 0.01, 0), (10_000, 0.01, -0.05),
    (np.nan, 0.01, 0.05),
])
def test_position_size_rejects_non_positive_inputs(account, risk, stop):
    with pytest.raises(ValueError):
        position_size(account, risk, stop)


def test_suggested_stop_is_twice_the_typical_move():
    assert suggested_stop(0.04) == pytest.approx(0.08)
    move, _, _ = typical_move_range(100.0, 0.02, horizon=4)
    assert suggested_stop(move) == pytest.approx(2 * move)
