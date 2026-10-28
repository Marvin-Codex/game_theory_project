"""Tests for mixed-strategy equilibria (Stage 5).

Hand-solved values are used wherever possible; Nashpy is only a cross-check.
"""

import numpy as np
import pytest

from game import TwoPlayerGame, price_war_game, promo_day_game
from mixed import all_nash_equilibria, mixed_equilibrium_2x2, expected_payoffs, is_pure
from equilibrium import pure_nash_equilibria


def make(A, B):
    A, B = np.array(A, float), np.array(B, float)
    n, m = A.shape
    return TwoPlayerGame("test", ["P1", "P2"],
                         [f"r{i}" for i in range(n)], [f"c{j}" for j in range(m)], A, B)


def mixed_only(game):
    return [(p, q) for p, q in all_nash_equilibria(game) if not (is_pure(p) and is_pure(q))]


# --- Project game: hand calculation p = q = 0.75, value 42.5 -----------------

def test_promo_day_closed_form():
    p, q = mixed_equilibrium_2x2(promo_day_game())
    assert p == pytest.approx(0.75)
    assert q == pytest.approx(0.75)


def test_promo_day_mixed_equilibrium_and_payoff():
    (p, q), = mixed_only(promo_day_game())
    assert p == pytest.approx([0.75, 0.25])
    assert q == pytest.approx([0.75, 0.25])
    assert expected_payoffs(promo_day_game(), p, q) == pytest.approx((42.5, 42.5))


def test_promo_day_has_three_equilibria():
    assert len(all_nash_equilibria(promo_day_game())) == 3


def test_price_war_has_no_mixed_equilibrium():
    g = price_war_game()
    assert mixed_equilibrium_2x2(g) is None
    assert len(all_nash_equilibria(g)) == 1


# --- Textbook games -----------------------------------------------------------

def test_matching_pennies():
    g = make([[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
    (p, q), = all_nash_equilibria(g)
    assert p == pytest.approx([0.5, 0.5])
    assert q == pytest.approx([0.5, 0.5])
    assert expected_payoffs(g, p, q) == pytest.approx((0, 0))


def test_battle_of_the_sexes_mixed():
    g = make([[2, 0], [0, 1]], [[1, 0], [0, 2]])
    p, q = mixed_equilibrium_2x2(g)
    assert p == pytest.approx(2 / 3)    # row plays first strategy w.p. 2/3
    assert q == pytest.approx(1 / 3)    # column plays first strategy w.p. 1/3
    (pv, qv), = mixed_only(g)
    assert expected_payoffs(g, pv, qv) == pytest.approx((2 / 3, 2 / 3))
    assert len(all_nash_equilibria(g)) == 3


def test_rock_paper_scissors():
    A = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    g = make(A, -np.array(A))
    (p, q), = all_nash_equilibria(g)
    assert p == pytest.approx([1 / 3] * 3)
    assert q == pytest.approx([1 / 3] * 3)
    assert expected_payoffs(g, p, q) == pytest.approx((0, 0))


# --- Properties ---------------------------------------------------------------

def test_pure_equilibria_agree_with_stage_4():
    for g in (price_war_game(), promo_day_game()):
        pure = sorted(
            (int(np.argmax(p)), int(np.argmax(q)))
            for p, q in all_nash_equilibria(g) if is_pure(p) and is_pure(q)
        )
        assert pure == pure_nash_equilibria(g)


def _is_equilibrium(game, p, q, tol=1e-7):
    va, vb = expected_payoffs(game, p, q)
    return (np.max(game.A @ q) <= va + tol) and (np.max(p @ game.B) <= vb + tol)


@pytest.mark.parametrize("seed", range(60))
def test_random_games_every_result_is_a_real_equilibrium(seed):
    rng = np.random.default_rng(seed)
    n, m = rng.integers(2, 5, size=2)
    g = make(rng.random((n, m)), rng.random((n, m)))
    results = all_nash_equilibria(g)
    assert len(results) % 2 == 1            # non-degenerate games have an odd number
    for p, q in results:
        assert p.sum() == pytest.approx(1) and q.sum() == pytest.approx(1)
        assert _is_equilibrium(g, p, q)


# --- Cross-check against Nashpy ----------------------------------------------

nash = pytest.importorskip("nashpy")


@pytest.mark.parametrize("seed", range(40))
def test_matches_nashpy_on_random_games(seed):
    rng = np.random.default_rng(1000 + seed)
    n, m = rng.integers(2, 4, size=2)
    A, B = rng.random((n, m)), rng.random((n, m))
    mine = all_nash_equilibria(make(A, B))
    theirs = list(nash.Game(A, B).support_enumeration())

    assert len(mine) == len(theirs)
    for p, q in mine:
        assert any(np.allclose(p, x) and np.allclose(q, y) for x, y in theirs)