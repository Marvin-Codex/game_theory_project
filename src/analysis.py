"""
analysis.py - Stage 3: payoff matrix analysis and best responses.

Best response (row player): given the column player plays strategy j, the row
player's best responses are the strategies i that maximise A[i, j].
Best response (column player): given the row player plays strategy i, the
column player's best responses are the strategies j that maximise B[i, j].

Ties are kept: a best response is a *set* of strategies, not always one.
"""

import numpy as np
import pandas as pd

from .game import TwoPlayerGame, price_war_game, promo_day_game

TOL = 1e-9  # tolerance for treating two payoffs as equal


def best_responses_row(game):
    """For each column strategy j, return the list of row best-response indices."""
    result = {}
    for j in range(len(game.col_strategies)):
        column = game.A[:, j]
        best = column.max()
        result[j] = [i for i in range(len(column)) if abs(column[i] - best) <= TOL]
    return result


def best_responses_col(game):
    """For each row strategy i, return the list of column best-response indices."""
    result = {}
    for i in range(len(game.row_strategies)):
        row = game.B[i, :]
        best = row.max()
        result[i] = [j for j in range(len(row)) if abs(row[j] - best) <= TOL]
    return result


def best_response_table(game):
    """Best responses of both players as readable strategy names."""
    row_br = best_responses_row(game)
    col_br = best_responses_col(game)

    p1, p2 = game.players
    rows = [
        {
            "Player": p1,
            f"If {p2} plays": game.col_strategies[j],
            "Best response": ", ".join(game.row_strategies[i] for i in row_br[j]),
        }
        for j in row_br
    ] + [
        {
            "Player": p2,
            f"If {p1} plays": game.row_strategies[i],
            "Best response": ", ".join(game.col_strategies[j] for j in col_br[i]),
        }
        for i in col_br
    ]
    return rows


def marked_dataframe(game):
    """Payoff matrix with '*' on each payoff that is a best response.

    A '*' on the first number means the row player is best-responding in that
    cell; a '*' on the second means the column player is.
    """
    row_br = best_responses_row(game)
    col_br = best_responses_col(game)
    fmt = TwoPlayerGame._fmt

    cells = []
    for i in range(len(game.row_strategies)):
        line = []
        for j in range(len(game.col_strategies)):
            a = fmt(game.A[i, j]) + ("*" if i in row_br[j] else "")
            b = fmt(game.B[i, j]) + ("*" if j in col_br[i] else "")
            line.append(f"({a}, {b})")
        cells.append(line)

    df = pd.DataFrame(cells, index=game.row_strategies, columns=game.col_strategies)
    df.index.name = f"{game.players[0]} \\ {game.players[1]}"
    return df


def report(game):
    """Print the best-response analysis of a game."""
    p1, p2 = game.players
    print(f"\nBest-response analysis: {game.name}")
    print("=" * (len(game.name) + 23))

    for entry in best_response_table(game):
        player = entry["Player"]
        other = p2 if player == p1 else p1
        given = entry[f"If {other} plays"]
        print(f"  {player}: if {other} plays {given}, best response is {entry['Best response']}")

    print(f"\nPayoff matrix with best responses marked (*), shown as ({p1}, {p2}):\n")
    print(marked_dataframe(game).to_string())
    print()


if __name__ == "__main__":
    report(price_war_game())
    report(promo_day_game())