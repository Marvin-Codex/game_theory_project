"""
game.py - Stage 2: representing and displaying a two-player game.

A finite two-player game in normal (strategic) form is stored as:
    - player names
    - a list of strategies for each player
    - two payoff matrices A (row player) and B (column player), where
      A[i, j] and B[i, j] are the payoffs when the row player plays
      strategy i and the column player plays strategy j.
"""

import json

import numpy as np
import pandas as pd


class TwoPlayerGame:
    """A finite two-player simultaneous-move game in normal form."""

    def __init__(self, name, players, row_strategies, col_strategies, A, B):
        try:
            A = np.array(A, dtype=float)
            B = np.array(B, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("Payoffs must be numbers arranged in a rectangular grid") from None

        if A.ndim != 2 or B.ndim != 2 or A.size == 0:
            raise ValueError("Payoffs must form a non-empty two-dimensional grid")
        if A.shape != B.shape:
            raise ValueError(f"Payoff matrices differ in shape: {A.shape} vs {B.shape}")
        if not (np.isfinite(A).all() and np.isfinite(B).all()):
            raise ValueError("Payoffs must be finite numbers (no NaN or infinity)")

        players = [str(p).strip() for p in players]
        if len(players) != 2:
            raise ValueError("Exactly two players are required")
        if not all(players) or players[0] == players[1]:
            raise ValueError("The two players need different, non-empty names")

        row_strategies = [str(s).strip() for s in row_strategies]
        col_strategies = [str(s).strip() for s in col_strategies]
        for who, names in ((players[0], row_strategies), (players[1], col_strategies)):
            if not names or not all(names):
                raise ValueError(f"{who} needs at least one named strategy")
            if len(set(names)) != len(names):
                raise ValueError(f"{who} has duplicate strategy names")
        if A.shape != (len(row_strategies), len(col_strategies)):
            raise ValueError(
                f"Payoff matrix shape {A.shape} does not match "
                f"{len(row_strategies)} row and {len(col_strategies)} column strategies"
            )

        self.name = name
        self.players = players
        self.row_strategies = row_strategies
        self.col_strategies = col_strategies
        self.A = A
        self.B = B

    @classmethod
    def from_cells(cls, name, players, row_strategies, col_strategies, cells):
        """Build a game from a nested list of (row payoff, column payoff) tuples."""
        A = [[cell[0] for cell in row] for row in cells]
        B = [[cell[1] for cell in row] for row in cells]
        return cls(name, players, row_strategies, col_strategies, A, B)

    @classmethod
    def from_dict(cls, data):
        """Build a game from a dict in the JSON format used by examples/.

        {"name": ..., "players": [p1, p2], "row_strategies": [...],
         "col_strategies": [...], "payoffs": [[[a, b], ...], ...]}
        where payoffs[i][j] = [row player's payoff, column player's payoff].
        """
        if not isinstance(data, dict):
            raise ValueError("A game description must be a JSON object")
        required = ("players", "row_strategies", "col_strategies", "payoffs")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError("Missing field(s): " + ", ".join(missing))

        payoffs = data["payoffs"]
        if not isinstance(payoffs, list) or not all(isinstance(row, list) for row in payoffs):
            raise ValueError("payoffs must be a list of rows")
        A, B = [], []
        for r, row in enumerate(payoffs):
            A.append([])
            B.append([])
            for c, cell in enumerate(row):
                if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                    raise ValueError(
                        f"payoffs[{r}][{c}] must be a [row payoff, column payoff] pair")
                A[-1].append(cell[0])
                B[-1].append(cell[1])
        return cls(data.get("name", "Custom game"), data["players"],
                   data["row_strategies"], data["col_strategies"], A, B)

    def to_dict(self):
        """The game as a plain dict (the inverse of from_dict)."""
        def num(x):
            return int(x) if float(x).is_integer() else float(x)

        return {
            "name": self.name,
            "players": list(self.players),
            "row_strategies": list(self.row_strategies),
            "col_strategies": list(self.col_strategies),
            "payoffs": [[[num(self.A[i, j]), num(self.B[i, j])]
                         for j in range(self.A.shape[1])]
                        for i in range(self.A.shape[0])],
        }

    @classmethod
    def load_json(cls, path):
        """Load a game from a JSON file. Raises ValueError with a readable message."""
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except OSError as err:
            raise ValueError(f"Cannot read {path}: {err.strerror or err}") from None
        except json.JSONDecodeError as err:
            raise ValueError(f"{path} is not valid JSON: {err}") from None
        try:
            return cls.from_dict(data)
        except ValueError as err:
            raise ValueError(f"{path}: {err}") from None

    def save_json(self, path):
        """Write the game to a JSON file."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    def payoff(self, i, j):
        """Return (row payoff, column payoff) for strategy indices (i, j)."""
        return self.A[i, j], self.B[i, j]

    @staticmethod
    def _fmt(x):
        return f"{x:g}"

    def to_dataframe(self):
        """Payoff matrix as a DataFrame whose cells read '(row, column)'."""
        cells = [
            [f"({self._fmt(self.A[i, j])}, {self._fmt(self.B[i, j])})"
             for j in range(len(self.col_strategies))]
            for i in range(len(self.row_strategies))
        ]
        df = pd.DataFrame(cells, index=self.row_strategies, columns=self.col_strategies)
        df.index.name = f"{self.players[0]} \\ {self.players[1]}"
        return df

    def display(self):
        """Print the game in a readable form."""
        print(f"\n{self.name}")
        print("=" * len(self.name))
        print(f"Players: {self.players[0]} (rows), {self.players[1]} (columns)")
        print(f"Payoffs are shown as ({self.players[0]}, {self.players[1]})\n")
        print(self.to_dataframe().to_string())
        print()


# ---------------------------------------------------------------------------
# Games for the project: two food shops near campus
# ---------------------------------------------------------------------------

def price_war_game():
    """Simultaneous lunch price war (daily profit, UGX '000).

    Price 5,000 (Keep) or 4,500 (Cut); cost 3,000 per plate; 100 students/day.
    Equal prices split students 50/50; a price cut wins 80 students from a
    rival that keeps its price.
    """
    return TwoPlayerGame.from_cells(
        name="Lunch Price War",
        players=["Shop A", "Shop B"],
        row_strategies=["Keep", "Cut"],
        col_strategies=["Keep", "Cut"],
        cells=[
            [(100, 100), (40, 120)],
            [(120, 40), (75, 75)],
        ],
    )


def promo_day_game():
    """Promotion-day game: each shop picks a day to run its lunch promotion.

    Tuesday is the busier day. Promoting on the same day splits the crowd,
    so each shop prefers to promote on a different day from its rival.
    """
    return TwoPlayerGame.from_cells(
        name="Promotion Day Game",
        players=["Shop A", "Shop B"],
        row_strategies=["Tue", "Thu"],
        col_strategies=["Tue", "Thu"],
        cells=[
            [(30, 30), (80, 50)],
            [(50, 80), (20, 20)],
        ],
    )


# ---------------------------------------------------------------------------
# Basic input: let the user enter their own game
# ---------------------------------------------------------------------------

def _ask_int(prompt, minimum=1):
    while True:
        try:
            value = int(input(prompt))
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"  Please enter a whole number >= {minimum}.")


def _ask_number(prompt):
    while True:
        try:
            value = float(input(prompt))
            if np.isfinite(value):
                return value
        except ValueError:
            pass
        print("  Please enter a finite number.")


def _ask_strategies(player, count, prefix):
    """Ask for `count` distinct strategy names for one player."""
    names = []
    for k in range(count):
        while True:
            name = input(f"  {player} strategy {k + 1}: ").strip() or f"{prefix}{k + 1}"
            if name not in names:
                names.append(name)
                break
            print("  That name is already used; choose a different one.")
    return names


def game_from_user():
    """Build a game interactively from keyboard input."""
    name = input("Game name: ").strip() or "Custom Game"
    p1 = input("Row player name: ").strip() or "Player 1"
    p2 = input("Column player name: ").strip() or "Player 2"
    if p2 == p1:
        p2 = f"{p2} (2)"

    n_rows = _ask_int(f"Number of strategies for {p1}: ")
    rows = _ask_strategies(p1, n_rows, "R")
    n_cols = _ask_int(f"Number of strategies for {p2}: ")
    cols = _ask_strategies(p2, n_cols, "C")

    A = np.zeros((n_rows, n_cols))
    B = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            A[i, j] = _ask_number(f"Payoff to {p1} when ({rows[i]}, {cols[j]}): ")
            B[i, j] = _ask_number(f"Payoff to {p2} when ({rows[i]}, {cols[j]}): ")

    return TwoPlayerGame(name, [p1, p2], rows, cols, A, B)


if __name__ == "__main__":
    price_war_game().display()
    promo_day_game().display()

    if input("Enter your own game? (y/n): ").strip().lower() == "y":
        game_from_user().display()