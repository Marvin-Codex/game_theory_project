"""Tests for scenario analysis (Stage 6).

Regime boundaries are derived by hand in the scenarios.py docstring; the tests
check the code lands on the correct side of each boundary.
"""

import numpy as np
import pytest

from game import price_war_game, promo_day_game
from scenarios import (
    build_price_war, build_promo_day, classify, analyse, scenario_table, sweep,
    regimes, promo_sweep, PRICE_WAR_SCENARIOS, PROMO_SCENARIOS,
)

PD = "Prisoner's Dilemma"


# --- The parameterised games reproduce the original ones ------------------------

def test_default_price_war_matches_original():
    new, old = build_price_war(), price_war_game()
    assert np.allclose(new.A, old.A) and np.allclose(new.B, old.B)


def test_default_promo_day_matches_original():
    new, old = build_promo_day(), promo_day_game()
    assert np.allclose(new.A, old.A) and np.allclose(new.B, old.B)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        build_price_war(price_cut=5200)          # cut above the normal price
    with pytest.raises(ValueError):
        build_price_war(win_share=0.3)           # cutter would win fewer than half
    with pytest.raises(ValueError):
        build_price_war(students=0)


# --- Base case -----------------------------------------------------------------

def test_base_price_war():
    a = analyse(build_price_war())
    assert a["type"] == PD
    assert a["pure_names"] == "(Cut, Cut)"
    assert a["delta"] == pytest.approx(4 / 9)


def test_base_promo_day():
    a = analyse(build_promo_day())
    assert a["type"] == "Anti-coordination game"
    assert a["pure_names"] == "(Tue, Thu); (Thu, Tue)"
    p, q = a["mixed"][0]
    assert p[0] == pytest.approx(0.75) and q[0] == pytest.approx(0.75)


# --- Price war: regime boundaries from hand algebra ----------------------------

@pytest.mark.parametrize("cost,expected", [
    (3000, PD),
    (3666, PD),                                   # just below 11/3
    (3667, "Coordination game"),                  # just above 11/3
    (4100, "Coordination game"),
    (4166, "Coordination game"),                  # just below 25/6
    (4167, "Dominant-strategy equilibrium"),      # just above 25/6
    (4400, "Dominant-strategy equilibrium"),
])
def test_cost_regimes(cost, expected):
    assert classify(build_price_war(cost=cost)) == expected


def test_high_cost_means_nobody_cuts():
    a = analyse(build_price_war(cost=4300))
    assert a["pure_names"] == "(Keep, Keep)"
    assert a["dominant"] == "Keep (both)"


def test_coordination_regime_has_two_pure_equilibria_and_a_mixed_one():
    a = analyse(build_price_war(cost=3800))
    assert a["pure_names"] == "(Keep, Keep); (Cut, Cut)"
    assert len(a["mixed"]) == 1


@pytest.mark.parametrize("cost", [2000, 2500, 3000, 3500, 3600])
def test_threshold_matches_closed_form(cost):
    c = cost / 1000
    expected = (110 - 30 * c) / (30 * (4.5 - c))     # (T-R)/(T-P) with defaults
    assert analyse(build_price_war(cost=cost))["delta"] == pytest.approx(expected)


def test_cooperation_gets_easier_as_cost_rises_within_dilemma_regime():
    df = sweep(build_price_war, "cost", range(2000, 3601, 200))
    deltas = df["delta*"].to_numpy()
    assert (df["Type"] == PD).all()
    assert np.all(np.diff(deltas) < 0)


# --- Price war: other parameters ------------------------------------------------

def test_smaller_discount_makes_cooperation_harder():
    a = analyse(build_price_war(price_cut=4800))
    assert a["type"] == PD
    assert a["delta"] == pytest.approx(44 / 54)       # (T-R)/(T-P), T=144, R=100, P=90


def test_weak_customer_reaction_removes_the_price_war():
    a = analyse(build_price_war(win_share=0.6))
    assert a["pure_names"] == "(Keep, Keep)"


def test_strong_customer_reaction():
    a = analyse(build_price_war(win_share=0.95))
    assert a["type"] == PD
    assert a["delta"] == pytest.approx(42.5 / 67.5)   # T=142.5, R=100, P=75


def test_deeper_discount_gives_coordination_game():
    assert classify(build_price_war(price_cut=4000)) == "Coordination game"


def test_scenario_table_covers_every_scenario():
    table = scenario_table(build_price_war, PRICE_WAR_SCENARIOS)
    assert list(table["Scenario"]) == list(PRICE_WAR_SCENARIOS)
    expected = [PD, "Coordination game", "Dominant-strategy equilibrium", PD,
                "Coordination game", "Dominant-strategy equilibrium", PD]
    assert list(table["Type"]) == expected


def test_regimes_groups_consecutive_rows():
    df = sweep(build_price_war, "cost", [3000, 3500, 3700, 3800, 4300])
    segs = regimes(df, "cost")
    assert [(s["from"], s["to"], s["Type"]) for s in segs] == [
        (3000, 3500, PD),
        (3700, 3800, "Coordination game"),
        (4300, 4300, "Dominant-strategy equilibrium"),
    ]


# --- Promotion day ----------------------------------------------------------------

def test_promo_scenarios_hand_solved():
    t = {name: analyse(build_promo_day(**p)) for name, p in PROMO_SCENARIOS.items()}

    # q = (thu_shared - tue_alone) / (tue_shared - tue_alone - thu_alone + thu_shared)
    assert t["Tuesday much busier"]["mixed"][0][1][0] == pytest.approx(100 / 105)
    assert t["Tuesday advantage vanishes"]["mixed"][0][1][0] == pytest.approx(0.5)
    assert t["Heavy clash penalty"]["mixed"][0][1][0] == pytest.approx(70 / 110)

    light = t["Light clash penalty"]
    assert light["type"] == "Dominant-strategy equilibrium"
    assert light["pure_names"] == "(Tue, Tue)"
    assert light["mixed"] == []


def test_heavier_clash_penalty_pushes_towards_clashing_less_on_tuesday():
    base = analyse(build_promo_day())["mixed"][0][0][0]
    heavy = analyse(build_promo_day(tue_shared=10, thu_shared=10))["mixed"][0][0][0]
    assert heavy < base          # 0.636 < 0.75


def test_promo_sweep_probability_of_tuesday_rises_with_its_advantage():
    # Boundary by hand: Tuesday becomes dominant once tue_shared = 0.375 v
    # exceeds thu_alone = 50, i.e. v > 133.3. Below it a mixed equilibrium exists.
    df = promo_sweep(range(80, 131, 10))
    probs = df["P(Tue)"].to_numpy()
    assert np.all(np.diff(probs) > 0)
    assert df["Clash probability"].between(0, 1).all()


def test_promo_sweep_tuesday_becomes_dominant_past_the_boundary():
    df = promo_sweep([130, 140]).set_index("tue_alone")
    assert df.at[130, "Type"] == "Anti-coordination game"
    assert df.at[140, "Type"] == "Dominant-strategy equilibrium"
    assert np.isnan(df.at[140, "P(Tue)"])