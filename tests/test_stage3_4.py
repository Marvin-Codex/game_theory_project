"""Tests for best responses, dominance and pure Nash (Stages 3 and 4).

Expected values come from hand calculation or standard textbook games.
Nashpy is used only as an independent cross-check, not as the implementation.
"""

import numpy as np
import pytest

from game import TwoPlayerGame, price_war_game, promo_day_game
from analysis import best_responses_row, best_responses_col
from equilibrium import (
    dominant_strategy,
    dominated_strategies,
    iterated_elimination,
    pure_nash_equilibria,
)


def make(cells, rows=None, cols=None):
    n, m = len(cells), len(cells[0])
    return TwoPlayerGame.from_cells(
        "test", ["P1", "P2"],
        rows or [f"r{i}" for i in range(n)],
        cols or [f"c{j}" for j in range(m)],
        cells,
    )


# --- Project games ----------------------------------------------------------

def test_price_war_best_responses():
    g = price_war_game()
    assert best_responses_row(g) == {0: [1], 1: [1]}
    assert best_responses_col(g) == {0: [1], 1: [1]}


def test_price_war_cut_strictly_dominant_for_both():
    g = price_war_game()
    assert dominant_strategy(g, 0) == 1
    assert dominant_strategy(g, 1) == 1


def test_price_war_unique_nash_is_cut_cut():
    assert pure_nash_equilibria(price_war_game()) == [(1, 1)]


def test_promo_day_has_no_dominant_strategy():
    g = promo_day_game()
    assert dominant_strategy(g, 0) is None
    assert dominant_strategy(g, 1) is None


def test_promo_day_two_pure_nash():
    assert pure_nash_equilibria(promo_day_game()) == [(0, 1), (1, 0)]


# --- Textbook games ---------------------------------------------------------

def test_prisoners_dilemma():
    g = make([[(-1, -1), (-3, 0)], [(0, -3), (-2, -2)]])
    assert pure_nash_equilibria(g) == [(1, 1)]
    assert dominant_strategy(g, 0) == 1


def test_battle_of_the_sexes_two_equilibria():
    g = make([[(2, 1), (0, 0)], [(0, 0), (1, 2)]])
    assert pure_nash_equilibria(g) == [(0, 0), (1, 1)]


def test_matching_pennies_has_no_pure_nash():
    g = make([[(1, -1), (-1, 1)], [(-1, 1), (1, -1)]])
    assert pure_nash_equilibria(g) == []


def test_stag_hunt_two_equilibria_no_dominance():
    g = make([[(4, 4), (0, 3)], [(3, 0), (3, 3)]])
    assert pure_nash_equilibria(g) == [(0, 0), (1, 1)]
    assert dominant_strategy(g, 0) is None


# --- Dominance details ------------------------------------------------------

def test_weak_but_not_strict_dominance():
    # Row 0 beats row 1 in column 0 and ties in column 1: weak dominance only.
    g = make([[(2, 0), (1, 0)], [(1, 0), (1, 0)]])
    assert dominant_strategy(g, 0, strict=True) is None
    assert dominant_strategy(g, 0, strict=False) == 0
    assert dominated_strategies(g, 0, strict=True) == {}
    assert dominated_strategies(g, 0, strict=False) == {1: [0]}


def test_iterated_elimination_solves_3x3():
    # Hand-solved: C is dominated by R (column player); then M and B are
    # dominated by T (row player); then R is dominated by L. Survivor: (T, L).
    g = make(
        [[(4, 3), (5, 1), (6, 2)],
         [(2, 1), (8, 4), (3, 6)],
         [(3, 0), (9, 6), (2, 8)]],
        rows=["T", "M", "B"], cols=["L", "C", "R"],
    )
    rows, cols, steps = iterated_elimination(g)
    assert rows == [0] and cols == [0]
    assert pure_nash_equilibria(g) == [(0, 0)]


def test_ties_are_all_reported_as_best_responses():
    g = make([[(1, 0), (2, 0)], [(1, 0), (0, 0)]])
    assert best_responses_row(g)[0] == [0, 1]


# --- Cross-check against Nashpy ---------------------------------------------

nash = pytest.importorskip("nashpy")


def _nashpy_pure(A, B):
    out = []
    for x, y in nash.Game(A, B).support_enumeration():
        if np.isclose(x.max(), 1) and np.isclose(y.max(), 1):
            out.append((int(x.argmax()), int(y.argmax())))
    return sorted(out)


@pytest.mark.parametrize("seed", range(40))
def test_matches_nashpy_on_random_games(seed):
    rng = np.random.default_rng(seed)
    n, m = rng.integers(2, 5, size=2)
    A = rng.random((n, m))      # continuous payoffs => non-degenerate games
    B = rng.random((n, m))
    g = TwoPlayerGame("random", ["P1", "P2"],
                      [f"r{i}" for i in range(n)], [f"c{j}" for j in range(m)], A, B)
    assert sorted(pure_nash_equilibria(g)) == _nashpy_pure(A, B)