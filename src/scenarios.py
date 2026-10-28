"""
scenarios.py - Stage 6: scenario analysis.

Builds the two games from underlying parameters (costs, prices, demand), so a
change in the real-world situation can be fed through the whole analysis:

    change a parameter -> rebuild payoffs -> recompute equilibria -> compare.

PRICE WAR (payoffs = daily profit, UGX '000)
    margin_keep = price_keep - cost,  margin_cut = price_cut - cost
    both keep:        each sells students/2 at margin_keep
    both cut:         each sells students/2 at margin_cut
    one cuts:         the cutter wins win_share * students at margin_cut,
                      the keeper keeps the rest at margin_keep
    With the default numbers this gives the payoffs 100 / 40 / 120 / 75.

PROMOTION DAY (daily profit, UGX '000)
    tue_alone / thu_alone    profit from promoting on that day alone
    tue_shared / thu_shared  profit from promoting on that day together with the rival
    With the defaults (80, 50, 30, 20) this is the original game.

Hand-derived regime boundaries for the price war (cost c in UGX '000,
other parameters at their defaults), used to test the code:
    c < 11/3 (3,667)          Cut beats Keep even when the rival keeps  -> Prisoner's Dilemma
    11/3 < c < 25/6 (4,167)   Cut only pays if the rival cuts           -> coordination game
    c > 25/6                  Keep dominates                            -> no price war
"""

import os

import numpy as np
import pandas as pd

from .analysis import TOL
from .equilibrium import dominant_strategy, pure_nash_equilibria
from .game import TwoPlayerGame
from .mixed import all_nash_equilibria, expected_payoffs, is_pure
from .repeated import grim_threshold, pd_parameters


# ---------------------------------------------------------------------------
# Parameterised games
# ---------------------------------------------------------------------------

def build_price_war(price_keep=5000, price_cut=4500, cost=3000, students=100, win_share=0.8):
    """Two shops choosing Keep or Cut. Payoffs in UGX '000 per day."""
    if not price_cut < price_keep:
        raise ValueError("price_cut must be below price_keep")
    if not 0.5 <= win_share <= 1.0:
        raise ValueError("win_share must be between 0.5 and 1")
    if students <= 0:
        raise ValueError("students must be positive")

    m_keep = (price_keep - cost) / 1000.0
    m_cut = (price_cut - cost) / 1000.0
    half = students / 2
    winner = students * win_share
    loser = students - winner

    keep_keep = round(half * m_keep, 6)
    cut_cut = round(half * m_cut, 6)
    cutter = round(winner * m_cut, 6)
    keeper = round(loser * m_keep, 6)

    return TwoPlayerGame.from_cells(
        "Lunch Price War", ["Shop A", "Shop B"], ["Keep", "Cut"], ["Keep", "Cut"],
        [[(keep_keep, keep_keep), (keeper, cutter)],
         [(cutter, keeper), (cut_cut, cut_cut)]],
    )


def build_promo_day(tue_alone=80, thu_alone=50, tue_shared=30, thu_shared=20):
    """Two shops choosing which day to run a promotion. Payoffs in UGX '000."""
    return TwoPlayerGame.from_cells(
        "Promotion Day Game", ["Shop A", "Shop B"], ["Tue", "Thu"], ["Tue", "Thu"],
        [[(tue_shared, tue_shared), (tue_alone, thu_alone)],
         [(thu_alone, tue_alone), (thu_shared, thu_shared)]],
    )


# ---------------------------------------------------------------------------
# Analysis of one game
# ---------------------------------------------------------------------------

def classify(game):
    """Name the strategic structure of a 2x2 game."""
    if game.A.shape != (2, 2):
        return "Larger game"
    try:
        pd_parameters(game)
        return "Prisoner's Dilemma"
    except ValueError:
        pass

    if all(dominant_strategy(game, p) is not None for p in (0, 1)):
        return "Dominant-strategy equilibrium"
    equilibria = pure_nash_equilibria(game)
    if (0, 0) in equilibria and (1, 1) in equilibria:
        return "Coordination game"
    if (0, 1) in equilibria and (1, 0) in equilibria:
        return "Anti-coordination game"
    return "Other"


def _mix_text(names, vec):
    return ", ".join(f"{n} {v:.0%}" for n, v in zip(names, vec) if v > TOL)


def analyse(game):
    """Collect the equilibrium facts about a game in one dict."""
    pure = pure_nash_equilibria(game)
    pure_names = "; ".join(
        f"({game.row_strategies[i]}, {game.col_strategies[j]})" for i, j in pure) or "none"
    pure_text = "; ".join(
        f"({game.row_strategies[i]}, {game.col_strategies[j]}) -> "
        f"({game.A[i, j]:g}, {game.B[i, j]:g})" for i, j in pure) or "none"

    doms = []
    for player, names in ((0, game.row_strategies), (1, game.col_strategies)):
        d = dominant_strategy(game, player)
        doms.append(names[d] if d is not None else None)
    if doms[0] and doms[0] == doms[1]:
        dominant_text = f"{doms[0]} (both)"
    elif any(doms):
        dominant_text = "; ".join(f"{game.players[k]}: {d}" for k, d in enumerate(doms) if d)
    else:
        dominant_text = "none"

    mixed = [(p, q) for p, q in all_nash_equilibria(game) if not (is_pure(p) and is_pure(q))]
    if mixed:
        p, q = mixed[0]
        a, b = expected_payoffs(game, p, q)
        mixed_text = (f"{game.players[0]} {_mix_text(game.row_strategies, p)}; "
                      f"{game.players[1]} {_mix_text(game.col_strategies, q)} -> ({a:.2f}, {b:.2f})")
    else:
        mixed_text = "none"

    kind = classify(game)
    delta = grim_threshold(game) if kind == "Prisoner's Dilemma" else float("nan")

    return {
        "type": kind, "dominant": dominant_text, "pure": pure, "pure_names": pure_names,
        "pure_text": pure_text, "mixed": mixed, "mixed_text": mixed_text, "delta": delta,
    }


# ---------------------------------------------------------------------------
# Scenario tables
# ---------------------------------------------------------------------------

PRICE_WAR_SCENARIOS = {
    "Base case": {},
    "Cost rises to 3,800": {"cost": 3800},
    "Cost rises to 4,300": {"cost": 4300},
    "Smaller discount (cut to 4,800)": {"price_cut": 4800},
    "Deeper discount (cut to 4,000)": {"price_cut": 4000},
    "Customers react weakly (60%)": {"win_share": 0.6},
    "Customers react strongly (95%)": {"win_share": 0.95},
}

PROMO_SCENARIOS = {
    "Base case": {},
    "Tuesday much busier": {"tue_alone": 120, "tue_shared": 45},
    "Tuesday advantage vanishes": {"tue_alone": 60, "thu_alone": 60,
                                   "tue_shared": 25, "thu_shared": 25},
    "Heavy clash penalty": {"tue_shared": 10, "thu_shared": 10},
    "Light clash penalty": {"tue_shared": 55, "thu_shared": 35},
}


def scenario_table(builder, scenarios):
    """One row per scenario with the equilibrium picture."""
    rows = []
    for name, params in scenarios.items():
        game = builder(**params)
        a = analyse(game)
        rows.append({
            "Scenario": name,
            "Type": a["type"],
            "Dominant strategy": a["dominant"],
            "Pure Nash -> payoffs": a["pure_text"],
            "Mixed equilibrium": a["mixed_text"],
            "delta*": "-" if np.isnan(a["delta"]) else f"{a['delta']:.3f}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parameter sweeps
# ---------------------------------------------------------------------------

def sweep(builder, param, values, **fixed):
    """Rebuild and analyse the game for each value of one parameter."""
    rows = []
    for value in values:
        a = analyse(builder(**{**fixed, param: value}))
        rows.append({param: value, "Type": a["type"], "Pure Nash": a["pure_names"],
                     "delta*": a["delta"], "mixed": a["mixed"]})
    return pd.DataFrame(rows)


def regimes(df, param):
    """Group consecutive sweep rows that share the same type and pure equilibria."""
    out, start = [], 0
    for i in range(1, len(df) + 1):
        if i == len(df) or (df.at[i, "Type"], df.at[i, "Pure Nash"]) != \
                (df.at[start, "Type"], df.at[start, "Pure Nash"]):
            seg = df.iloc[start:i]
            out.append({
                "from": seg[param].iloc[0], "to": seg[param].iloc[-1],
                "Type": seg["Type"].iloc[0], "Pure Nash": seg["Pure Nash"].iloc[0],
                "delta first": seg["delta*"].iloc[0], "delta last": seg["delta*"].iloc[-1],
            })
            start = i
    return out


def print_regimes(df, param, label):
    print(f"\n  {label}")
    for r in regimes(df, param):
        span = f"{r['from']:g}" if r["from"] == r["to"] else f"{r['from']:g} to {r['to']:g}"
        line = f"    {span}: {r['Type']}, Nash {r['Pure Nash']}"
        if not np.isnan(r["delta first"]):
            line += f", delta* {r['delta first']:.3f} -> {r['delta last']:.3f}"
        print(line)


def promo_sweep(values):
    """Tuesday's advantage: how the mixed equilibrium moves as tue_alone grows
    (tue_shared rises in proportion, keeping the clash penalty at 37.5%)."""
    rows = []
    for v in values:
        game = build_promo_day(tue_alone=v, tue_shared=0.375 * v)
        a = analyse(game)
        if a["mixed"]:
            p, q = a["mixed"][0]
            pa, _ = expected_payoffs(game, p, q)
            rows.append({"tue_alone": v, "Type": a["type"], "P(Tue)": p[0],
                         "Clash probability": p[0] * q[0] + p[1] * q[1],
                         "Mixed payoff": pa})
        else:
            rows.append({"tue_alone": v, "Type": a["type"], "P(Tue)": float("nan"),
                         "Clash probability": float("nan"), "Mixed payoff": float("nan")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _show(df):
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(df.to_string(index=False))


def report(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("\nSCENARIO ANALYSIS: PRICE WAR")
    print("============================")
    table = scenario_table(build_price_war, PRICE_WAR_SCENARIOS)
    _show(table)
    table.to_csv(os.path.join(output_dir, "scenarios_price_war.csv"), index=False)

    costs = list(range(2000, 4501, 100))
    cost_df = sweep(build_price_war, "cost", costs)
    print_regimes(cost_df, "cost", "Sweep: cost per plate (UGX), cut price fixed at 4,500")
    cost_df.drop(columns="mixed").to_csv(os.path.join(output_dir, "sweep_cost.csv"), index=False)

    shares = [round(0.5 + 0.05 * k, 2) for k in range(11)]
    share_df = sweep(build_price_war, "win_share", shares)
    print_regimes(share_df, "win_share", "Sweep: share of students won by a price cut")
    share_df.drop(columns="mixed").to_csv(os.path.join(output_dir, "sweep_win_share.csv"), index=False)

    print("\n\nSCENARIO ANALYSIS: PROMOTION DAY")
    print("================================")
    ptable = scenario_table(build_promo_day, PROMO_SCENARIOS)
    _show(ptable)
    ptable.to_csv(os.path.join(output_dir, "scenarios_promo_day.csv"), index=False)

    print("\n  Sweep: Tuesday's advantage (profit alone on Tuesday; Thursday alone = 50)")
    psweep = promo_sweep(range(50, 151, 10))
    _show(psweep.round(3))
    psweep.to_csv(os.path.join(output_dir, "sweep_promo_tuesday.csv"), index=False)
    print()


if __name__ == "__main__":
    report()