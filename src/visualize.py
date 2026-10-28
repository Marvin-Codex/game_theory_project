"""
visualize.py - Stage 6: figures for the report.

Four figures, each saved as a PNG in figures/:
    fig_payoff_matrices.png        payoff matrices, best responses and Nash equilibria
    fig_best_responses.png         best-response diagram (where the mixed equilibrium comes from)
    fig_cooperation_threshold.png  how patient the shops must be, as cost changes
    fig_tuesday_sweep.png          the mixed equilibrium as Tuesday gets busier

Colour: Shop A = blue, Shop B = orange (validated pair); one-hue blue ramp for
magnitude; text always in neutral ink, never in a series colour; gridlines are
solid hairlines. Every figure has a table twin in outputs/ (see scenarios.py).
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from .analysis import best_responses_col, best_responses_row
from .equilibrium import pure_nash_equilibria
from .game import price_war_game, promo_day_game
from .mixed import all_nash_equilibria, is_pure
from .repeated import grim_threshold
from .scenarios import build_price_war, build_promo_day, classify, promo_sweep, sweep

# Palette (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND_1 = "#f0efec"
BAND_2 = "#e8e7e2"
SERIES_1 = "#2a78d6"   # Shop A
SERIES_2 = "#eb6834"   # Shop B
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5"]   # sequential steps 100-400

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _style(ax, grid=True):
    """Quiet axes: hairline solid grid, no top/right spines, neutral ink."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=AXIS)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)


def _title(ax, title, subtitle=None):
    lines = subtitle.count("\n") + 1 if subtitle else 0
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="semibold",
                 pad=10 + 13 * lines)
    if subtitle:
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def _save(fig, filename):
    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(filename, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _boundary(make, lo, hi, tol=0.5):
    """Bisect for the parameter value where the type of game changes."""
    kind = classify(make(lo))
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if classify(make(mid)) == kind:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Figure 1: payoff matrices
# ---------------------------------------------------------------------------

def plot_payoff_matrices(games, filename):
    """Each cell shows (row payoff, column payoff). Shade = joint profit,
    bold = best response, ring = Nash equilibrium."""
    n = len(games)
    fig = plt.figure(figsize=(5.0 * n + 1.3, 5.3))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.04], wspace=0.32)
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    vmax = max(float((g.A + g.B).max()) for g in games)
    norm = Normalize(vmin=0, vmax=vmax)

    for k, game in enumerate(games):
        ax = fig.add_subplot(gs[0, k])
        nr, nc = game.A.shape
        row_br, col_br = best_responses_row(game), best_responses_col(game)
        nash = set(pure_nash_equilibria(game))

        for i in range(nr):
            for j in range(nc):
                a, b = game.A[i, j], game.B[i, j]
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=cmap(norm(a + b)),
                                       edgecolor=SURFACE, linewidth=2.5))
                cx, cy = j + 0.5, i + 0.55
                ax.text(cx - 0.02, cy, f"{a:g},", ha="right", va="center", fontsize=14,
                        color=INK, fontweight="bold" if i in row_br[j] else "normal")
                ax.text(cx + 0.05, cy, f"{b:g}", ha="left", va="center", fontsize=14,
                        color=INK, fontweight="bold" if j in col_br[i] else "normal")
                if (i, j) in nash:
                    ax.add_patch(Rectangle((j + 0.07, i + 0.07), 0.86, 0.86, fill=False,
                                           edgecolor=INK, linewidth=2.2))
                    ax.text(cx, i + 0.22, "Nash equilibrium", ha="center", va="center",
                            fontsize=8.5, color=INK2)

        ax.set_xlim(0, nc)
        ax.set_ylim(nr, 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(nc) + 0.5)
        ax.set_xticklabels(game.col_strategies, fontsize=11, color=INK)
        ax.set_yticks(np.arange(nr) + 0.5)
        ax.set_yticklabels(game.row_strategies, fontsize=11, color=INK)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(length=0, pad=6)
        ax.set_xlabel(f"{game.players[1]} (columns)", fontsize=10, color=INK2, labelpad=8)
        ax.set_ylabel(f"{game.players[0]} (rows)", fontsize=10, color=INK2, labelpad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{game.name}: {classify(game)}", loc="left", fontsize=12,
                     color=INK, fontweight="semibold", pad=16)

    cax = fig.add_subplot(gs[0, n])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK2, labelsize=9, length=3, color=AXIS)
    cbar.set_label("Joint profit of both shops, UGX '000 per day", color=INK2, fontsize=9)

    fig.text(0.5, 0.01,
             "Cells show (row payoff, column payoff) in UGX '000 per day.   "
             "Bold = best response.   Ring = Nash equilibrium.",
             ha="center", color=INK2, fontsize=9)
    _save(fig, filename)


# ---------------------------------------------------------------------------
# Figure 2: best-response diagram
# ---------------------------------------------------------------------------

def _best_response_path(M):
    """Best-response correspondence of a player with own payoff matrix M
    (own strategies on rows, opponent's on columns), as a polyline of
    (opponent probability of strategy 0, own probability of strategy 0)."""
    d0 = M[0, 1] - M[1, 1]      # advantage of own strategy 0 when opponent plays strategy 1
    d1 = M[0, 0] - M[1, 0]      # ... when opponent plays strategy 0
    if abs(d0) < 1e-9 and abs(d1) < 1e-9:
        return None
    if d0 >= -1e-9 and d1 >= -1e-9:
        return [(0, 1), (1, 1)]
    if d0 <= 1e-9 and d1 <= 1e-9:
        return [(0, 0), (1, 0)]
    t = d0 / (d0 - d1)          # opponent mix that makes the player indifferent
    left = 1 if d0 > 0 else 0
    return [(0, left), (t, left), (t, 1 - left), (1, 1 - left)]


def _equilibrium_points(game):
    """(x, y, label) for every Nash equilibrium, x = P(column plays first),
    y = P(row plays first)."""
    points = []
    for p, q in all_nash_equilibria(game):
        if is_pure(p) and is_pure(q):
            label = f"({game.row_strategies[int(np.argmax(p))]}, {game.col_strategies[int(np.argmax(q))]})"
        else:
            a, b = (name.split()[-1] for name in game.players)
            label = f"Mixed: {a} {p[0]:.0%},\n{b} {q[0]:.0%}"
        points.append((float(q[0]), float(p[0]), label))
    return points


def plot_best_responses(games, filename):
    n = len(games)
    fig, axes = plt.subplots(1, n, figsize=(5.3 * n, 6.0))
    axes = np.atleast_1d(axes)
    pct = FuncFormatter(lambda v, _: f"{v:.0%}")

    for ax, game in zip(axes, games):
        _style(ax)
        p1, p2 = game.players
        row_path = _best_response_path(game.A)
        col_path = _best_response_path(game.B.T)

        if row_path:
            xs, ys = zip(*row_path)
            ax.plot(xs, ys, color=SERIES_1, linewidth=2.2, solid_capstyle="round",
                    solid_joinstyle="round", zorder=3)
        if col_path:
            ys, xs = zip(*col_path)             # column player's own prob is the x coordinate
            ax.plot(xs, ys, color=SERIES_2, linewidth=2.2, solid_capstyle="round",
                    solid_joinstyle="round", zorder=3)

        for x, y, label in _equilibrium_points(game):
            ax.plot([x], [y], marker="o", markersize=9, markerfacecolor=INK,
                    markeredgecolor=SURFACE, markeredgewidth=1.8, linestyle="none", zorder=5)
            mixed = label.startswith("Mixed")
            dx = 0.04 if (mixed or x < 0.5) else -0.04
            dy = 0.05 if (mixed or y < 0.5) else -0.05
            ax.text(x + dx, y + dy, label, ha="left" if dx > 0 else "right",
                    va="bottom" if dy > 0 else "top", fontsize=9.5, color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(pct)
        ax.yaxis.set_major_formatter(pct)
        ax.set_xlabel(f"Probability {p2} plays {game.col_strategies[0]}", fontsize=10)
        ax.set_ylabel(f"Probability {p1} plays {game.row_strategies[0]}", fontsize=10)
        ax.set_title(f"{game.name}: {classify(game)}", loc="left", fontsize=12,
                     color=INK, fontweight="semibold", pad=10)

    p1, p2 = games[0].players
    handles = [
        Line2D([], [], color=SERIES_1, linewidth=2.2, label=f"{p1}'s best response"),
        Line2D([], [], color=SERIES_2, linewidth=2.2, label=f"{p2}'s best response"),
        Line2D([], [], marker="o", markersize=8, markerfacecolor=INK, markeredgecolor=SURFACE,
               linestyle="none", label="Nash equilibrium (where the lines meet)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, filename)


# ---------------------------------------------------------------------------
# Figure 3: how patient must the shops be?
# ---------------------------------------------------------------------------

def plot_cooperation_threshold(filename, cost_range=(2000, 4500)):
    make = lambda c: build_price_war(cost=c)
    b1 = _boundary(make, 3000, 3800)       # Prisoner's Dilemma -> coordination game
    b2 = _boundary(make, 3800, 4400)       # coordination game -> Keep dominates
    lo, hi = cost_range

    costs = np.append(np.arange(lo, b1, 20), b1 - 0.5)
    df = sweep(build_price_war, "cost", costs)
    dilemma = df[df["Type"] == "Prisoner's Dilemma"]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    _style(ax)
    ax.axvspan(b1, b2, color=BAND_1, zorder=0, linewidth=0)
    ax.axvspan(b2, hi, color=BAND_2, zorder=0, linewidth=0)

    ax.plot(dilemma["cost"], dilemma["delta*"], color=SERIES_1, linewidth=2.2,
            solid_capstyle="round", zorder=3)

    d0 = grim_threshold(build_price_war())
    ax.plot([3000], [d0], marker="o", markersize=9, markerfacecolor=SERIES_1,
            markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=5)
    ax.annotate(f"Base case (cost 3,000)\n$\\delta^*$ = {d0:.3f}", xy=(3000, d0),
                xytext=(3150, 0.62), fontsize=9.5, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8, shrinkA=0, shrinkB=6))

    for x0, x1, text in ((lo, b1, "Prisoner's Dilemma: price war"),
                         (b1, b2, "Coordination game"),
                         (b2, hi, "Keep\ndominates")):
        ax.text((x0 + x1) / 2, 0.97, text, ha="center", va="top", fontsize=9, color=INK2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(lo, hi + 1, 500))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Cost per plate (UGX)", fontsize=10)
    ax.set_ylabel("Minimum discount factor $\\delta^*$ for cooperation", fontsize=10)
    for edge in (b1, b2):
        ax.text(edge + 15, 0.03, f"{edge:,.0f}", ha="left", va="bottom", fontsize=8.5, color=INK2)
    _title(ax, "How patient must the shops be to keep prices up?",
           "Fatter margins make undercutting more tempting. Cut price fixed at UGX 4,500.\n"
           "Grim trigger and tit-for-tat need the same threshold here.")
    _save(fig, filename)
    return b1, b2


# ---------------------------------------------------------------------------
# Figure 4: Tuesday's advantage
# ---------------------------------------------------------------------------

def plot_tuesday_sweep(filename, values=None):
    values = list(values if values is not None else range(50, 151, 2))
    df = promo_sweep(values)
    make = lambda v: build_promo_day(tue_alone=v, tue_shared=0.375 * v)
    boundary = _boundary(make, 100, 150, tol=0.1)
    x_lo, x_hi = min(values), max(values)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.8, 7.6), sharex=True,
                                   gridspec_kw={"hspace": 0.16})
    for ax in (ax1, ax2):
        _style(ax)
        ax.axvspan(boundary, x_hi, color=BAND_1, zorder=0, linewidth=0)

    # (a) probabilities - same unit, one axis
    ax1.plot(df["tue_alone"], df["P(Tue)"], color=SERIES_1, linewidth=2.2, solid_capstyle="round", zorder=3)
    ax1.plot(df["tue_alone"], df["Clash probability"], color=SERIES_2, linewidth=2.2,
             solid_capstyle="round", zorder=3)
    ax1.text(98, 0.865 + 0.04, "Each shop promotes on Tuesday", ha="right", va="bottom", fontsize=9, color=INK2)
    ax1.text(102, 0.766 - 0.045, "Both shops pick the same day", ha="left", va="top", fontsize=9, color=INK2)
    ax1.text((boundary + x_hi) / 2, 0.45, "Tuesday\ndominant:\nno mixed\nequilibrium",
             ha="center", va="center", fontsize=8.5, color=INK2)
    ax1.set_ylim(0, 1.02)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_ylabel("Probability in the mixed equilibrium", fontsize=10)
    ax1.legend(handles=[
        Line2D([], [], color=SERIES_1, linewidth=2.2, label="Each shop promotes on Tuesday"),
        Line2D([], [], color=SERIES_2, linewidth=2.2, label="Both shops pick the same day (clash)"),
    ], loc="lower right", bbox_to_anchor=(0.78, 0.04), frameon=False, fontsize=9, labelcolor=INK2)
    _title(ax1, "A busier Tuesday pulls both shops towards Tuesday, and towards clashing",
           "Mixed equilibrium of the promotion-day game. Thursday pays 50 alone and 20 in a clash.\n"
           "A Tuesday clash leaves each shop 37.5% of its solo Tuesday profit.")

    # (b) expected profit - a different measure, so its own panel
    ax2.plot(df["tue_alone"], df["Mixed payoff"], color=SERIES_1, linewidth=2.2, solid_capstyle="round", zorder=3)
    ax2.set_ylim(0, 60)
    ax2.set_ylabel("Expected profit per shop\n(UGX '000 per day)", fontsize=10)
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_xlabel("Profit from promoting alone on Tuesday (UGX '000 per day)", fontsize=10)
    ax2.text(x_lo + 2, 57, "Mixed-equilibrium profit, same for both shops", ha="left", va="top",
             fontsize=9, color=INK2)

    ax1.axvline(boundary, color=AXIS, linewidth=1, zorder=1)
    ax2.axvline(boundary, color=AXIS, linewidth=1, zorder=1)
    _save(fig, filename)
    return boundary


# ---------------------------------------------------------------------------

def make_all(folder="figures"):
    plot_payoff_matrices([price_war_game(), promo_day_game()],
                         os.path.join(folder, "fig_payoff_matrices.png"))
    plot_best_responses([price_war_game(), promo_day_game()],
                        os.path.join(folder, "fig_best_responses.png"))
    b1, b2 = plot_cooperation_threshold(os.path.join(folder, "fig_cooperation_threshold.png"))
    boundary = plot_tuesday_sweep(os.path.join(folder, "fig_tuesday_sweep.png"))
    return {"cost_boundaries": (b1, b2), "tuesday_boundary": boundary}


if __name__ == "__main__":
    info = make_all()
    print("Saved four figures to figures/")
    print(f"Regime changes: cost {info['cost_boundaries'][0]:,.0f} and "
          f"{info['cost_boundaries'][1]:,.0f}; Tuesday profit {info['tuesday_boundary']:.1f}")