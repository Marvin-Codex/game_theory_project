"""
gametree.py - Stage 5: sequential games as game trees.

A finite perfect-information game is stored as a NetworkX directed tree:
    - decision nodes record which player moves (0 or 1),
    - leaf nodes record the payoff pair (player 0, player 1),
    - each edge is labelled with the action that leads along it.

Solution concept: backward induction, which gives the subgame-perfect
equilibrium. Working from the leaves up, the player who moves at a node picks
the action that gives them the highest payoff in the already-solved subtree
below. If a player is indifferent between actions, all of them are marked
optimal and the first one listed is used to continue the calculation.
"""

import os

import networkx as nx
import matplotlib

matplotlib.use("Agg")  # draw to files, no display needed
import matplotlib.pyplot as plt

from .analysis import TOL
from .game import price_war_game, promo_day_game
from .mixed import all_nash_equilibria, expected_payoffs


class GameTree:
    """A two-player perfect-information game tree."""

    def __init__(self, name, players):
        if len(players) != 2:
            raise ValueError("Exactly two players are required")
        self.name = name
        self.players = list(players)
        self.graph = nx.DiGraph()
        self.root = None
        self._next_id = 0

    def _new_node(self, **attrs):
        node = self._next_id
        self._next_id += 1
        self.graph.add_node(node, **attrs)
        return node

    def _attach(self, node, parent, action):
        g = self.graph
        if parent is None:
            if self.root is not None:
                raise ValueError("The tree already has a root")
            self.root = node
            g.nodes[node]["history"] = ()
            return
        if g.nodes[parent]["kind"] != "decision":
            raise ValueError("Cannot add children to a leaf")
        if any(d["action"] == action for _, _, d in g.out_edges(parent, data=True)):
            raise ValueError(f"Action {action!r} already used at this node")
        g.add_edge(parent, node, action=action)
        g.nodes[node]["history"] = g.nodes[parent]["history"] + (action,)

    def add_decision(self, player, parent=None, action=None):
        """Add a decision node for `player` (0 or 1). Returns its node id."""
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        node = self._new_node(kind="decision", player=player)
        self._attach(node, parent, action)
        return node

    def add_leaf(self, payoffs, parent, action):
        """Add a leaf with payoffs (player 0, player 1). Returns its node id."""
        if len(payoffs) != 2:
            raise ValueError("A leaf needs exactly two payoffs")
        node = self._new_node(kind="leaf", payoffs=tuple(float(x) for x in payoffs))
        self._attach(node, parent, action)
        return node

    @classmethod
    def from_normal_form(cls, game, name=None):
        """Turn a simultaneous game into a sequential one: the row player moves
        first and the column player observes that move before choosing."""
        tree = cls(name or f"{game.name} (sequential)", game.players)
        root = tree.add_decision(0)
        for i, row_action in enumerate(game.row_strategies):
            node = tree.add_decision(1, parent=root, action=row_action)
            for j, col_action in enumerate(game.col_strategies):
                tree.add_leaf(game.payoff(i, j), parent=node, action=col_action)
        return tree


# ---------------------------------------------------------------------------
# Backward induction
# ---------------------------------------------------------------------------

def backward_induction(tree):
    """Solve the tree by backward induction.

    Marks optimal edges (edge attribute 'optimal') and stores 'best_actions'
    and 'value' on every node. Returns a dict with:
        payoffs : payoff pair at the equilibrium outcome
        path    : list of (player, action) along the equilibrium path
        ties    : decision nodes where the mover was indifferent
    """
    g = tree.graph
    for _, _, data in g.edges(data=True):
        data["optimal"] = False
    ties = []

    def solve(node):
        data = g.nodes[node]
        if data["kind"] == "leaf":
            data["value"] = data["payoffs"]
            return data["value"]

        mover = data["player"]
        options = [(e["action"], child, solve(child))
                   for _, child, e in g.out_edges(node, data=True)]
        best_payoff = max(value[mover] for _, _, value in options)
        best = [o for o in options if o[2][mover] >= best_payoff - TOL]

        for _, child, _ in best:
            g.edges[node, child]["optimal"] = True
        if len(best) > 1:
            ties.append(node)
        data["best_actions"] = [action for action, _, _ in best]
        data["value"] = best[0][2]
        return data["value"]

    value = solve(tree.root)

    path, node = [], tree.root
    while g.nodes[node]["kind"] == "decision":
        action = g.nodes[node]["best_actions"][0]
        path.append((g.nodes[node]["player"], action))
        node = next(c for _, c, e in g.out_edges(node, data=True) if e["action"] == action)

    return {"payoffs": value, "path": path, "ties": ties}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(tree):
    """Print the subgame-perfect equilibrium of a tree."""
    result = backward_induction(tree)
    g = tree.graph

    title = f"Backward induction: {tree.name}"
    print(f"\n{title}\n{'=' * len(title)}")
    print("  Optimal action at every decision node:")
    for node in nx.dfs_preorder_nodes(g, tree.root):
        data = g.nodes[node]
        if data["kind"] != "decision":
            continue
        who = tree.players[data["player"]]
        where = "at the start" if not data["history"] else "after " + ", ".join(data["history"])
        print(f"    {who} {where}: {' or '.join(data['best_actions'])}")

    steps = ", then ".join(f"{tree.players[p]} plays {a}" for p, a in result["path"])
    a, b = result["payoffs"]
    print(f"  Equilibrium path: {steps}")
    print(f"  Payoffs: ({a:g}, {b:g})")
    if result["ties"]:
        print("  Note: some players were indifferent; the first listed action was used.")
    print()
    return result


def compare_with_simultaneous(game):
    """Compare simultaneous play (all Nash equilibria) with sequential play."""
    tree = GameTree.from_normal_form(game)
    result = backward_induction(tree)
    first, second = game.players

    title = f"Simultaneous vs sequential: {game.name}"
    print(f"\n{title}\n{'=' * len(title)}")
    print("  Simultaneous moves, Nash equilibria:")
    for p, q in all_nash_equilibria(game):
        a, b = expected_payoffs(game, p, q)
        kind = "pure" if max(p) > 1 - TOL and max(q) > 1 - TOL else "mixed"
        print(f"    {kind}: ({a:.4g}, {b:.4g})")
    a, b = result["payoffs"]
    print(f"  {first} moves first, {second} responds: ({a:g}, {b:g})")
    print()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw(tree, filename, solved=True):
    """Draw the tree to an image file. Optimal edges are highlighted if solved."""
    g = tree.graph
    if solved:
        backward_induction(tree)

    leaves = [n for n in nx.dfs_preorder_nodes(g, tree.root) if g.out_degree(n) == 0]
    x = {leaf: float(i) for i, leaf in enumerate(leaves)}

    def place(node):
        if node not in x:
            xs = [place(child) for child in g.successors(node)]
            x[node] = sum(xs) / len(xs)
        return x[node]

    place(tree.root)
    depth = nx.shortest_path_length(g, tree.root)
    max_depth = max(depth.values())
    y = {n: -float(d) for n, d in depth.items()}

    fig, ax = plt.subplots(figsize=(max(6.5, 2.4 * len(leaves)), 2.4 * max_depth + 1.8))

    for u, v, data in g.edges(data=True):
        highlighted = solved and data.get("optimal", False)
        ax.plot([x[u], x[v]], [y[u], y[v]],
                color="#1f6f8b" if highlighted else "#b0b0b0",
                linewidth=3.2 if highlighted else 1.3, zorder=1)
        ax.text((x[u] + x[v]) / 2, (y[u] + y[v]) / 2, data["action"],
                ha="center", va="center", fontsize=10, zorder=3,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

    for node, data in g.nodes(data=True):
        if data["kind"] == "decision":
            ax.text(x[node], y[node], tree.players[data["player"]],
                    ha="center", va="center", fontsize=11, fontweight="bold", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.45", fc="#e8f0f5", ec="#1f6f8b", lw=1.5))
        else:
            a, b = data["payoffs"]
            ax.text(x[node], y[node], f"({a:g}, {b:g})",
                    ha="center", va="center", fontsize=11, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#555555", lw=1.2))

    ax.set_xlim(-0.7, len(leaves) - 0.3)
    ax.set_ylim(-max_depth - 0.5, 0.5)
    ax.axis("off")
    ax.set_title(f"{tree.name}\npayoffs: ({tree.players[0]}, {tree.players[1]})", fontsize=12)

    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for game in (price_war_game(), promo_day_game()):
        tree = GameTree.from_normal_form(game)
        report(tree)
        compare_with_simultaneous(game)
        slug = game.name.lower().replace(" ", "_")
        draw(tree, f"figures/tree_{slug}.png")
        print(f"Saved figures/tree_{slug}.png")