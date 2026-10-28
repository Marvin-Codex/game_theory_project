"""Tests for the figures (Stage 6).

Pixels are not compared. The tests check that every figure is produced, and that
the regime boundaries printed on the figures match the hand-derived values.
"""

import os

import pytest

from game import price_war_game, promo_day_game
from visualize import (
    plot_payoff_matrices, plot_best_responses, plot_cooperation_threshold,
    plot_tuesday_sweep, _best_response_path, _boundary, _equilibrium_points,
)
from scenarios import build_price_war, build_promo_day


def _is_image(path):
    return os.path.getsize(path) > 5000


def test_payoff_matrix_figure(tmp_path):
    target = str(tmp_path / "m.png")
    plot_payoff_matrices([price_war_game(), promo_day_game()], target)
    assert _is_image(target)


def test_best_response_figure(tmp_path):
    target = str(tmp_path / "br.png")
    plot_best_responses([price_war_game(), promo_day_game()], target)
    assert _is_image(target)


def test_cooperation_threshold_figure_reports_hand_derived_boundaries(tmp_path):
    target = str(tmp_path / "c.png")
    b1, b2 = plot_cooperation_threshold(target)
    assert _is_image(target)
    assert b1 == pytest.approx(11000 / 3, abs=1)     # 3,667 = 11/3 in UGX '000
    assert b2 == pytest.approx(25000 / 6, abs=1)     # 4,167 = 25/6


def test_tuesday_figure_reports_hand_derived_boundary(tmp_path):
    target = str(tmp_path / "t.png")
    boundary = plot_tuesday_sweep(target)
    assert _is_image(target)
    assert boundary == pytest.approx(50 / 0.375, abs=0.5)    # 133.3


def test_boundary_helper_on_a_known_switch():
    make = lambda c: build_price_war(cost=c)
    assert _boundary(make, 3000, 3800, tol=0.01) == pytest.approx(11000 / 3, abs=0.05)


# --- The geometry behind the best-response diagram ---------------------------------

def test_promo_day_best_response_paths():
    g = promo_day_game()
    # Row player: plays Tue (1) while the column player is unlikely to play Tue,
    # Thu (0) after the 75% switch point.
    assert _best_response_path(g.A) == [(0, 1), (pytest.approx(0.75), 1), (pytest.approx(0.75), 0), (1, 0)]
    assert _best_response_path(g.B.T) == [(0, 1), (pytest.approx(0.75), 1), (pytest.approx(0.75), 0), (1, 0)]


def test_price_war_best_response_is_constant_cut():
    g = price_war_game()
    assert _best_response_path(g.A) == [(0, 0), (1, 0)]       # Cut whatever the rival does
    assert _best_response_path(g.B.T) == [(0, 0), (1, 0)]


def test_equilibrium_points_for_promo_day():
    pts = {(round(x, 3), round(y, 3)) for x, y, _ in _equilibrium_points(promo_day_game())}
    assert pts == {(0.0, 1.0), (1.0, 0.0), (0.75, 0.75)}


def test_equilibrium_points_for_price_war():
    pts = {(round(x, 3), round(y, 3)) for x, y, _ in _equilibrium_points(price_war_game())}
    assert pts == {(0.0, 0.0)}


def test_dominant_game_best_response_is_a_single_line():
    # Tuesday dominant for the row player: always plays Tue, so the path is constant at 1.
    g = build_promo_day(tue_shared=55, thu_shared=35)
    assert _best_response_path(g.A) == [(0, 1), (1, 1)]