"""Tests for the repeated game (Stage 5).

The threshold formulas are checked two ways: against hand arithmetic, and
against brute-force simulation of the discounted payoff streams.
"""

import pytest

from game import price_war_game, promo_day_game
from repeated import (
    all_cooperate, all_defect, tit_for_tat, grim_trigger,
    defect_once_then_cooperate,
    play, tournament, pd_parameters, grim_threshold, tft_thresholds,
    tft_threshold, cooperation_sustainable, finite_horizon_spe,
)

G = price_war_game()


# --- Parameters and thresholds (hand arithmetic) -------------------------------

def test_pd_parameters():
    assert pd_parameters(G) == {"R": 100, "T": 120, "P": 75, "S": 40}


def test_promo_day_is_not_a_prisoners_dilemma():
    with pytest.raises(ValueError):
        pd_parameters(promo_day_game())


def test_thresholds_match_hand_calculation():
    assert grim_threshold(G) == pytest.approx(20 / 45)       # (T-R)/(T-P)
    one_shot, forever = tft_thresholds(G)
    assert one_shot == pytest.approx(20 / 60)                # (T-R)/(R-S)
    assert forever == pytest.approx(20 / 45)
    assert tft_threshold(G) == pytest.approx(4 / 9)


# --- Strategies ---------------------------------------------------------------

def test_tit_for_tat_copies_last_move():
    assert tit_for_tat([], []) == 0
    assert tit_for_tat([0], [1]) == 1
    assert tit_for_tat([0, 1], [1, 0]) == 0


def test_grim_trigger_never_forgives():
    assert grim_trigger([0, 0], [0, 0]) == 0
    assert grim_trigger([0, 0, 0], [0, 1, 0]) == 1


def test_play_known_payoff_streams():
    # Grim vs Always Cut at delta=0.5: round 0 earns S=40, then P=75 forever.
    _, _, (grim_value, cut_value) = play(G, grim_trigger, all_defect, 2000, 0.5)
    assert grim_value == pytest.approx(40 + 0.5 * 75 / 0.5)
    assert cut_value == pytest.approx(120 + 0.5 * 75 / 0.5)


def test_one_shot_deviation_against_tit_for_tat_matches_formula():
    # T + d S + d^2 R / (1-d)
    d = 0.6
    _, _, (value, _) = play(G, defect_once_then_cooperate, tit_for_tat, 2000, d)
    assert value == pytest.approx(120 + d * 40 + d * d * 100 / (1 - d))


# --- Thresholds against simulation --------------------------------------------

@pytest.mark.parametrize("strategy", [grim_trigger, tit_for_tat])
def test_cooperation_flips_at_four_ninths(strategy):
    below, _, _ = cooperation_sustainable(G, strategy, 0.44)
    above, _, _ = cooperation_sustainable(G, strategy, 0.45)
    assert not below
    assert above


@pytest.mark.parametrize("delta", [0.45, 0.6, 0.8, 0.95])
@pytest.mark.parametrize("strategy", [grim_trigger, tit_for_tat])
def test_no_deviation_pays_above_threshold(strategy, delta):
    ok, coop, devs = cooperation_sustainable(G, strategy, delta)
    assert ok
    assert all(coop >= v for v in devs.values())


@pytest.mark.parametrize("delta", [0.05, 0.2, 0.4])
def test_defection_wins_below_threshold(delta):
    ok, _, _ = cooperation_sustainable(G, grim_trigger, delta)
    assert not ok


def test_tit_for_tat_one_shot_deviation_threshold_is_one_third():
    # Cut once then Keep against tit-for-tat: unprofitable iff delta >= 1/3.
    def gain(d):
        _, _, (coop, _) = play(G, tit_for_tat, tit_for_tat, 2000, d)
        _, _, (dev, _) = play(G, defect_once_then_cooperate, tit_for_tat, 2000, d)
        return dev - coop

    assert gain(0.32) > 0
    assert gain(0.34) < 0


# --- Tournament -----------------------------------------------------------------

def test_tournament_known_entries():
    t = tournament(G, rounds=20)
    assert t.loc["Always Keep", "Always Keep"] == pytest.approx(100)
    assert t.loc["Always Keep", "Always Cut"] == pytest.approx(40)
    assert t.loc["Always Cut", "Always Keep"] == pytest.approx(120)
    assert t.loc["Always Cut", "Always Cut"] == pytest.approx(75)
    assert t.loc["Tit-for-tat", "Always Cut"] == pytest.approx((40 + 19 * 75) / 20)
    assert t.loc["Always Cut", "Tit-for-tat"] == pytest.approx((120 + 19 * 75) / 20)
    assert t.loc["Tit-for-tat", "Grim trigger"] == pytest.approx(100)


# --- Finitely repeated game ---------------------------------------------------

@pytest.mark.parametrize("rounds", [1, 2, 5, 10])
def test_finite_horizon_unravels_to_cut_cut(rounds):
    plan, total = finite_horizon_spe(G, rounds)
    assert len(plan) == rounds
    for p, q in plan:
        assert p.argmax() == 1 and q.argmax() == 1          # Cut, Cut every round
    assert total == pytest.approx((75 * rounds, 75 * rounds))


def test_finite_horizon_with_discounting():
    plan, total = finite_horizon_spe(G, 4, delta=0.5)
    expected = 75 * (1 - 0.5 ** 4) / (1 - 0.5)
    assert total == pytest.approx((expected, expected))


def test_finite_horizon_undetermined_when_stage_game_has_several_equilibria():
    assert finite_horizon_spe(promo_day_game(), 3) is None