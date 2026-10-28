"""Tests for the game tree and backward induction (Stage 5)."""

import os

import pytest

from game import price_war_game, promo_day_game
from gametree import GameTree, backward_induction, draw


def entry_game():
    """Textbook entry game: Entrant stays Out (0, 2) or Enters; the Incumbent
    then Fights (-1, -1) or Accommodates (1, 1)."""
    t = GameTree("entry", ["Entrant", "Incumbent"])
    root = t.add_decision(0)
    out = t.add_decision(1, parent=root, action="Out")
    t.add_leaf((0, 2), out, "Fight")
    t.add_leaf((0, 2), out, "Accommodate")
    enter = t.add_decision(1, parent=root, action="Enter")
    t.add_leaf((-1, -1), enter, "Fight")
    t.add_leaf((1, 1), enter, "Accommodate")
    return t


def centipede():
    """Three-stage game: P0, then P1, then P0 may each Stop or Continue."""
    t = GameTree("centipede", ["P0", "P1"])
    root = t.add_decision(0)
    t.add_leaf((1, 0), root, "Stop")
    n1 = t.add_decision(1, parent=root, action="Continue")
    t.add_leaf((0, 2), n1, "Stop")
    n2 = t.add_decision(0, parent=n1, action="Continue")
    t.add_leaf((3, 1), n2, "Stop")
    t.add_leaf((2, 4), n2, "Continue")
    return t


# --- Structure --------------------------------------------------------------

def test_from_normal_form_structure():
    tree = GameTree.from_normal_form(price_war_game())
    assert tree.graph.number_of_nodes() == 1 + 2 + 4
    assert tree.graph.number_of_edges() == 2 + 4


def test_duplicate_action_rejected():
    t = GameTree("t", ["A", "B"])
    root = t.add_decision(0)
    t.add_leaf((1, 1), root, "x")
    with pytest.raises(ValueError):
        t.add_leaf((2, 2), root, "x")


# --- Backward induction -------------------------------------------------------

def test_price_war_sequential_is_cut_cut():
    result = backward_induction(GameTree.from_normal_form(price_war_game()))
    assert result["path"] == [(0, "Cut"), (1, "Cut")]
    assert result["payoffs"] == (75, 75)       # moving first gives no advantage


def test_promo_day_first_mover_advantage():
    tree = GameTree.from_normal_form(promo_day_game())
    result = backward_induction(tree)
    assert result["path"] == [(0, "Tue"), (1, "Thu")]
    assert result["payoffs"] == (80, 50)       # beats A's 42.5 in the mixed equilibrium
    # B's plan: avoid whatever day A picked
    g = tree.graph
    plans = {g.nodes[n]["history"]: g.nodes[n]["best_actions"]
             for n in g if g.nodes[n]["kind"] == "decision" and g.nodes[n]["player"] == 1}
    assert plans == {("Tue",): ["Thu"], ("Thu",): ["Tue"]}


def test_entry_game():
    result = backward_induction(entry_game())
    assert result["path"] == [(0, "Enter"), (1, "Accommodate")]
    assert result["payoffs"] == (1, 1)
    assert len(result["ties"]) == 1            # incumbent indifferent after Out


def test_centipede_unravels_to_stop_immediately():
    # Hand solution: P0 stops at the last node (3 > 2), so P1 stops (2 > 1),
    # so P0 stops at the start (1 > 0).
    result = backward_induction(centipede())
    assert result["path"] == [(0, "Stop")]
    assert result["payoffs"] == (1, 0)


# --- Drawing ----------------------------------------------------------------

def test_draw_creates_image(tmp_path):
    target = str(tmp_path / "tree.png")
    draw(GameTree.from_normal_form(promo_day_game()), target)
    assert os.path.getsize(target) > 1000