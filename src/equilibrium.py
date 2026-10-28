"""
equilibrium.py - Stage 4: dominant strategies and pure-strategy Nash equilibria.

Definitions (for a player choosing among own strategies s, t against every
opponent strategy):
    s strictly dominates t   if payoff(s) >  payoff(t) against EVERY opponent strategy.
    s weakly dominates t     if payoff(s) >= payoff(t) against every opponent strategy,
                             and > against at least one.
    A dominant strategy      is one that dominates every other strategy of the player.
    A dominated strategy     is one that some other strategy dominates.

Only dominance by a *pure* strategy is checked here (dominance by a mixture
of strategies is not covered).

A pure-strategy Nash equilibrium is a cell (i, j) where i is a best response
to j AND j is a best response to i.
"""

import numpy as np

from .analysis import TOL, best_responses_row, best_responses_col
from .game import price_war_game, promo_day_game


# ---------------------------------------------------------------------------
# Dominance
# ---------------------------------------------------------------------------

def _row_payoffs(game, player):
    """Payoff matrix of `player` (0 = row, 1 = column) with OWN strategies on rows."""
    return game.A if player == 0 else game.B.T


def _strategy_names(game, player):
    return game.row_strategies if player == 0 else game.col_strategies


def dominates(P, s, t, strict=True):
    """Does own strategy s dominate own strategy t in payoff matrix P?"""
    diff = P[s, :] - P[t, :]
    if strict:
        return bool(np.all(diff > TOL))
    return bool(np.all(diff >= -TOL) and np.any(diff > TOL))


def dominated_strategies(game, player, strict=True):
    """Return {dominated strategy index: [indices of strategies that dominate it]}."""
    P = _row_payoffs(game, player)
    n = P.shape[0]
    result = {}
    for t in range(n):
        dominators = [s for s in range(n) if s != t and dominates(P, s, t, strict)]
        if dominators:
            result[t] = dominators
    return result


def dominant_strategy(game, player, strict=True):
    """Index of the player's dominant strategy, or None if there isn't one."""
    P = _row_payoffs(game, player)
    n = P.shape[0]
    for s in range(n):
        if all(dominates(P, s, t, strict) for t in range(n) if t != s):
            return s
    return None


def iterated_elimination(game):
    """Iterated elimination of strictly dominated strategies (IESDS).

    Returns (remaining row indices, remaining column indices, list of steps),
    where each step is (player, eliminated strategy name).
    """
    rows = list(range(len(game.row_strategies)))
    cols = list(range(len(game.col_strategies)))
    steps = []

    changed = True
    while changed:
        changed = False

        # Row player's payoffs on the surviving sub-game
        A = game.A[np.ix_(rows, cols)]
        for t in range(len(rows)):
            if len(rows) > 1 and any(dominates(A, s, t) for s in range(len(rows)) if s != t):
                steps.append((game.players[0], game.row_strategies[rows[t]]))
                rows.pop(t)
                changed = True
                break
        if changed:
            continue

        # Column player's payoffs (own strategies on rows)
        Bt = game.B[np.ix_(rows, cols)].T
        for t in range(len(cols)):
            if len(cols) > 1 and any(dominates(Bt, s, t) for s in range(len(cols)) if s != t):
                steps.append((game.players[1], game.col_strategies[cols[t]]))
                cols.pop(t)
                changed = True
                break

    return rows, cols, steps


# ---------------------------------------------------------------------------
# Pure-strategy Nash equilibria
# ---------------------------------------------------------------------------

def pure_nash_equilibria(game):
    """Return a list of (i, j) index pairs that are pure-strategy Nash equilibria."""
    row_br = best_responses_row(game)   # row_br[j] = row best responses to column j
    col_br = best_responses_col(game)   # col_br[i] = column best responses to row i
    return [
        (i, j)
        for i in range(len(game.row_strategies))
        for j in range(len(game.col_strategies))
        if i in row_br[j] and j in col_br[i]
    ]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(game):
    """Print the dominance and Nash analysis of a game."""
    title = f"Equilibrium analysis: {game.name}"
    print(f"\n{title}\n{'=' * len(title)}")

    for player in (0, 1):
        names = _strategy_names(game, player)
        who = game.players[player]
        for strict, label in ((True, "strictly"), (False, "weakly")):
            dom = dominant_strategy(game, player, strict)
            if dom is not None:
                print(f"  {who}: {names[dom]} is {label} dominant")
                break
        else:
            print(f"  {who}: no dominant strategy")

        dominated = dominated_strategies(game, player, strict=True)
        for t, by in dominated.items():
            print(f"  {who}: {names[t]} is strictly dominated by {', '.join(names[s] for s in by)}")

    rows, cols, steps = iterated_elimination(game)
    if steps:
        order = "; ".join(f"{p} drops {s}" for p, s in steps)
        print(f"  Iterated elimination: {order}")
    else:
        print("  Iterated elimination: nothing to eliminate")

    equilibria = pure_nash_equilibria(game)
    if equilibria:
        print("  Pure Nash equilibria:")
        for i, j in equilibria:
            a, b = game.payoff(i, j)
            print(f"    ({game.row_strategies[i]}, {game.col_strategies[j]}) "
                  f"with payoffs ({a:g}, {b:g})")
    else:
        print("  No pure-strategy Nash equilibrium (check mixed strategies, Stage 5)")
    print()


if __name__ == "__main__":
    report(price_war_game())
    report(promo_day_game())