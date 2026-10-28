"""
mixed.py - Stage 5: mixed-strategy Nash equilibria.

Method: support enumeration based on the indifference principle.

In a mixed equilibrium, every pure strategy a player uses with positive
probability must earn the same expected payoff against the opponent's mix, and
no unused strategy may earn more. For a pair of supports (I for the row player,
J for the column player) of equal size k we solve two small linear systems:

    column mix q on J  such that every row in I earns the same value v:
        A[I, J] q = v,   sum(q) = 1
    row mix p on I     such that every column in J earns the same value w:
        p B[I, J] = w,   sum(p) = 1

and accept the pair if p, q > 0 and no strategy outside the support does
strictly better. Supports of size 1 give the pure equilibria, so this finds
pure and mixed equilibria together.

Limitation: supports whose linear system is singular (degenerate games) are
skipped, and only equal-sized supports are tried, which is sufficient for
non-degenerate games.
"""

from itertools import combinations

import numpy as np

from .game import TwoPlayerGame, promo_day_game, price_war_game
from .analysis import TOL


def expected_payoffs(game, p, q):
    """Expected payoffs (row player, column player) when the players mix p and q."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(p @ game.A @ q), float(p @ game.B @ q)


def _solve_indifference(payoff_block):
    """Solve  M x = v 1,  sum(x) = 1  for (x, v). Returns None if singular."""
    k = payoff_block.shape[0]
    M = np.zeros((k + 1, k + 1))
    M[:k, :k] = payoff_block
    M[:k, k] = -1.0
    M[k, :k] = 1.0
    rhs = np.zeros(k + 1)
    rhs[k] = 1.0
    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None
    return sol[:k], sol[k]


def mixed_equilibrium_2x2(game):
    """Closed-form fully mixed equilibrium of a 2x2 game, or None.

    The row player mixes (p, 1-p) so the COLUMN player is indifferent, and the
    column player mixes (q, 1-q) so the ROW player is indifferent:

        q = (A11 - A01) / (A00 - A01 - A10 + A11)
        p = (B11 - B10) / (B00 - B10 - B01 + B11)

    Returns (p, q): the probabilities of the first row / first column strategy.
    """
    if game.A.shape != (2, 2):
        raise ValueError("mixed_equilibrium_2x2 needs a 2x2 game")
    A, B = game.A, game.B

    den_q = A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1]
    den_p = B[0, 0] - B[1, 0] - B[0, 1] + B[1, 1]
    if abs(den_q) <= TOL or abs(den_p) <= TOL:
        return None

    q = (A[1, 1] - A[0, 1]) / den_q
    p = (B[1, 1] - B[1, 0]) / den_p
    if not (TOL < p < 1 - TOL and TOL < q < 1 - TOL):
        return None
    return float(p), float(q)


def all_nash_equilibria(game):
    """All Nash equilibria (pure and mixed) by support enumeration.

    Returns a list of (p, q) pairs of probability vectors, sorted by support
    size and then lexicographically so output order is stable.
    """
    A, B = game.A, game.B
    n, m = A.shape
    found = []

    for k in range(1, min(n, m) + 1):
        for I in combinations(range(n), k):
            for J in combinations(range(m), k):
                I_, J_ = list(I), list(J)

                q_sol = _solve_indifference(A[np.ix_(I_, J_)])
                p_sol = _solve_indifference(B[np.ix_(I_, J_)].T)
                if q_sol is None or p_sol is None:
                    continue
                (q_J, v), (p_I, w) = q_sol, p_sol

                if np.any(q_J <= TOL) or np.any(p_I <= TOL):
                    continue

                p = np.zeros(n)
                q = np.zeros(m)
                p[I_] = p_I
                q[J_] = q_J

                # No profitable deviation outside the supports
                if np.max(A @ q) > v + 1e-7 or np.max(p @ B) > w + 1e-7:
                    continue

                found.append((p, q))

    found.sort(key=lambda pq: (np.count_nonzero(pq[0]), tuple(-pq[0]), tuple(-pq[1])))
    return found


def is_pure(vec):
    return bool(np.isclose(np.max(vec), 1.0))


def _describe_mix(names, vec):
    parts = [f"{names[i]} {vec[i]:.4g}" for i in range(len(vec)) if vec[i] > TOL]
    return ", ".join(parts)


def report(game):
    """Print every Nash equilibrium of the game, pure and mixed."""
    title = f"Nash equilibria (pure and mixed): {game.name}"
    print(f"\n{title}\n{'=' * len(title)}")

    p1, p2 = game.players
    for idx, (p, q) in enumerate(all_nash_equilibria(game), start=1):
        kind = "pure" if is_pure(p) and is_pure(q) else "mixed"
        a, b = expected_payoffs(game, p, q)
        print(f"  Equilibrium {idx} ({kind})")
        print(f"    {p1}: {_describe_mix(game.row_strategies, p)}")
        print(f"    {p2}: {_describe_mix(game.col_strategies, q)}")
        print(f"    Expected payoffs: ({a:.4g}, {b:.4g})")

        if kind == "mixed":
            print(f"    Check: {p1}'s pure strategies against {p2}'s mix")
            for i, name in enumerate(game.row_strategies):
                print(f"      {name}: {float(game.A[i] @ q):.4g}")
            print(f"    Check: {p2}'s pure strategies against {p1}'s mix")
            for j, name in enumerate(game.col_strategies):
                print(f"      {name}: {float(p @ game.B[:, j]):.4g}")
    print()


if __name__ == "__main__":
    report(promo_day_game())
    report(price_war_game())