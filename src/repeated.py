"""
repeated.py - Stage 5: repeated games.

The two shops meet in the same stage game every day. A strategy is a function
of the history: strategy(my_history, opponent_history) -> action index.
Payoffs are discounted: round t is worth delta**t, where delta can be read as
how much tomorrow's profit counts relative to today's (or the probability that
the rivalry is still going tomorrow).

Notation for the price war, a Prisoner's Dilemma (cooperate = Keep, defect = Cut):
    R = reward         (both cooperate)            = 100
    T = temptation     (I defect, they cooperate)  = 120
    P = punishment     (both defect)               =  75
    S = sucker         (I cooperate, they defect)  =  40
with T > R > P > S.

Infinitely repeated game, one-shot deviation analysis:

    Grim trigger (cooperate until the rival defects, then defect forever)
        cooperate forever:  R / (1 - d)
        defect now:         T + d P / (1 - d)
        cooperation holds iff   d >= (T - R) / (T - P)

    Tit-for-tat (copy the rival's last move), two relevant deviations:
        defect once, then cooperate:  T + d S + d^2 R / (1 - d)
            holds iff   d >= (T - R) / (R - S)
        defect forever:               T + d P / (1 - d)
            holds iff   d >= (T - R) / (T - P)
        tit-for-tat is an equilibrium iff d is at least the larger of the two.

Finitely repeated game: if the stage game has a unique Nash equilibrium, the
unique subgame-perfect equilibrium repeats it every round (backward induction
unravels cooperation from the last round).
"""

import numpy as np
import pandas as pd

from .game import TwoPlayerGame, price_war_game, promo_day_game
from .mixed import all_nash_equilibria, expected_payoffs, is_pure

COOPERATE, DEFECT = 0, 1   # Keep, Cut in price_war_game


# ---------------------------------------------------------------------------
# Strategies: f(my_history, opponent_history) -> action index
# ---------------------------------------------------------------------------

def all_cooperate(my_hist, opp_hist):
    return COOPERATE


def all_defect(my_hist, opp_hist):
    return DEFECT


def tit_for_tat(my_hist, opp_hist):
    """Cooperate first, then copy the rival's previous move."""
    return COOPERATE if not opp_hist else opp_hist[-1]


def grim_trigger(my_hist, opp_hist):
    """Cooperate until the rival defects once, then defect forever."""
    return DEFECT if DEFECT in opp_hist else COOPERATE


def defect_once_then_cooperate(my_hist, opp_hist):
    """A deviation: cut on day one, then go back to keeping the price."""
    return DEFECT if not my_hist else COOPERATE


def alternate(my_hist, opp_hist):
    """A deviation: cut, keep, cut, keep, ..."""
    return DEFECT if len(my_hist) % 2 == 0 else COOPERATE


STRATEGIES = {
    "Always Keep": all_cooperate,
    "Always Cut": all_defect,
    "Tit-for-tat": tit_for_tat,
    "Grim trigger": grim_trigger,
}

DEVIATIONS = {
    "Always Cut": all_defect,
    "Cut once, then Keep": defect_once_then_cooperate,
    "Alternate Cut/Keep": alternate,
}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def play(game, strategy_a, strategy_b, rounds, delta=1.0):
    """Play the repeated game. Returns (history A, history B, discounted payoffs)."""
    hist_a, hist_b = [], []
    total = np.zeros(2)
    for t in range(rounds):
        a = strategy_a(hist_a, hist_b)
        b = strategy_b(hist_b, hist_a)
        pa, pb = game.payoff(a, b)
        total += (delta ** t) * np.array([pa, pb])
        hist_a.append(a)
        hist_b.append(b)
    return hist_a, hist_b, (float(total[0]), float(total[1]))


def tournament(game, rounds=20, strategies=None):
    """Round-robin: average payoff per round of the ROW strategy against the
    COLUMN strategy, with no discounting. Last column is the row's mean."""
    strategies = strategies or STRATEGIES
    names = list(strategies)
    table = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            _, _, (pa, _) = play(game, strategies[a], strategies[b], rounds)
            table.loc[a, b] = pa / rounds
    table["Mean"] = table.mean(axis=1)
    table.index.name = "Row \\ Column"
    return table


# ---------------------------------------------------------------------------
# Theory: payoffs and discount-factor thresholds
# ---------------------------------------------------------------------------

def pd_parameters(game, cooperate=COOPERATE, defect=DEFECT):
    """Return {R, T, P, S} if the game is a symmetric Prisoner's Dilemma."""
    if game.A.shape != (2, 2):
        raise ValueError("Prisoner's Dilemma analysis needs a 2x2 game")
    if not np.allclose(game.B, game.A.T):
        raise ValueError("Game is not symmetric")
    A = game.A
    R, S = A[cooperate, cooperate], A[cooperate, defect]
    T, P = A[defect, cooperate], A[defect, defect]
    if not (T > R > P > S):
        raise ValueError(f"Not a Prisoner's Dilemma: need T > R > P > S, got "
                         f"T={T:g}, R={R:g}, P={P:g}, S={S:g}")
    return {"R": float(R), "T": float(T), "P": float(P), "S": float(S)}


def grim_threshold(game):
    """Smallest discount factor at which grim trigger sustains cooperation."""
    q = pd_parameters(game)
    return (q["T"] - q["R"]) / (q["T"] - q["P"])


def tft_thresholds(game):
    """(vs 'defect once then cooperate', vs 'defect forever') thresholds for
    tit-for-tat. Tit-for-tat is an equilibrium iff delta >= max of the two."""
    q = pd_parameters(game)
    return ((q["T"] - q["R"]) / (q["R"] - q["S"]),
            (q["T"] - q["R"]) / (q["T"] - q["P"]))


def tft_threshold(game):
    return max(tft_thresholds(game))


def cooperation_sustainable(game, strategy, delta, rounds=2000):
    """Check by simulation whether both players following `strategy` beats each
    deviation in DEVIATIONS. Returns (sustainable, value of cooperating,
    {deviation name: value of deviating})."""
    _, _, (coop, _) = play(game, strategy, strategy, rounds, delta)
    deviations = {}
    for name, dev in DEVIATIONS.items():
        _, _, (value, _) = play(game, dev, strategy, rounds, delta)
        deviations[name] = value
    return coop >= max(deviations.values()) - 1e-9, coop, deviations


# ---------------------------------------------------------------------------
# Finitely repeated game: backward induction
# ---------------------------------------------------------------------------

def finite_horizon_spe(game, rounds, delta=1.0):
    """Backward induction on the `rounds`-times repeated game.

    Works from the last round back. The payoffs of each round's game are the
    stage payoffs plus the discounted value of continuing from the already
    solved later rounds. If any round's game has more than one Nash
    equilibrium, backward induction alone does not pin down play and None is
    returned. Otherwise returns (plan, total payoffs), where plan is a list of
    (p, q) mixes in round order.
    """
    continuation = np.zeros(2)
    plan = []
    for _ in range(rounds):
        augmented = TwoPlayerGame(
            game.name, game.players, game.row_strategies, game.col_strategies,
            game.A + delta * continuation[0], game.B + delta * continuation[1],
        )
        equilibria = all_nash_equilibria(augmented)
        if len(equilibria) != 1:
            return None
        p, q = equilibria[0]
        plan.append((p, q))
        continuation = np.array(expected_payoffs(augmented, p, q))
    plan.reverse()
    return plan, (float(continuation[0]), float(continuation[1]))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(game, rounds=20):
    title = f"Repeated game analysis: {game.name}"
    print(f"\n{title}\n{'=' * len(title)}")

    try:
        q = pd_parameters(game)
    except ValueError as err:
        print(f"  Not applicable: {err}")
        print()
        return
    R, T, P, S = q["R"], q["T"], q["P"], q["S"]
    print(f"  Prisoner's Dilemma payoffs: T={T:g}  R={R:g}  P={P:g}  S={S:g}")

    # Infinite horizon
    g = grim_threshold(game)
    one_shot, forever = tft_thresholds(game)
    print("\n  Infinitely repeated game (delta = weight on tomorrow's profit)")
    print(f"    Grim trigger:  delta >= (T-R)/(T-P) = {T - R:g}/{T - P:g} = {g:.4f}")
    print(f"    Tit-for-tat:   delta >= (T-R)/(R-S) = {T - R:g}/{R - S:g} = {one_shot:.4f}  (cut once, then keep)")
    print(f"                   delta >= (T-R)/(T-P) = {T - R:g}/{T - P:g} = {forever:.4f}  (cut forever)")
    print(f"                   so tit-for-tat needs delta >= {max(one_shot, forever):.4f}")

    print("\n  Check by simulation, grim trigger: value of cooperating vs best deviation")
    print("    delta   cooperate   best deviation   sustainable")
    for delta in sorted({0.30, 0.40, g, 0.50, 0.90}):
        ok, coop, devs = cooperation_sustainable(game, grim_trigger, delta)
        print(f"    {delta:.4f}  {coop:9.2f}   {max(devs.values()):14.2f}   {'yes' if ok else 'no'}")

    # Finite horizon
    n = 10
    result = finite_horizon_spe(game, n)
    print(f"\n  Finitely repeated game, {n} rounds, backward induction")
    if result is None:
        print("    Stage game has several equilibria, so backward induction does not fix play.")
    else:
        plan, total = result
        p, q_ = plan[0]
        names = (game.row_strategies[int(np.argmax(p))], game.col_strategies[int(np.argmax(q_))])
        same = all(is_pure(a) and is_pure(b) and int(np.argmax(a)) == int(np.argmax(p))
                   and int(np.argmax(b)) == int(np.argmax(q_)) for a, b in plan)
        print(f"    Unique equilibrium: ({names[0]}, {names[1]}) in {'every round' if same else 'round 1'}; "
              f"total payoffs ({total[0]:g}, {total[1]:g})")
        print("    Cooperation unravels from the last round, so a known end date kills it.")

    # Tournament
    print(f"\n  Round-robin, {rounds} rounds, average payoff per round (row strategy):\n")
    print(tournament(game, rounds).round(2).to_string())
    print()


if __name__ == "__main__":
    report(price_war_game())
    report(promo_day_game())