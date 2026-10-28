# Game Theory Analysis System

A comprehensive Python-based application for modeling, analyzing, and visualizing strategic interactions between decision-makers. This project implements a complete game-theoretic analysis toolkit covering normal-form games, extensive-form games, and repeated games with full support for dominance analysis, Nash equilibrium computation (pure and mixed strategies), and scenario-based comparative analysis.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Mathematical Formulations](#mathematical-formulations)
- [Results and Analysis](#results-and-analysis)
- [Testing and Validation](#testing-and-validation)
- [Dependencies](#dependencies)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This system provides a robust framework for analyzing strategic decision-making problems using game theory. It was developed to demonstrate the practical application of game-theoretic concepts in business competition scenarios, specifically focusing on two competing food shops making pricing and promotional decisions on a university campus. The implementation emphasizes algorithmic correctness by building core game theory solvers from scratch while using established libraries for validation and visualization.

## Problem Statement

**Strategic Context:** Two food shops (Shop A and Shop B) operate near a university campus and compete for students' business. They face two distinct strategic scenarios:

### 1. Lunch Price War (Simultaneous Pricing Game)
- **Decision:** Each shop chooses between keeping their lunch price at UGX 5,000 or cutting it to UGX 4,500
- **Market Dynamics:**
  - Cost per plate: UGX 3,000
  - Total students per day: 100
  - If both charge the same price, they split the market equally (50 each)
  - If one cuts price while the other keeps it, the price-cutter captures 80% of students (80 vs 20)
- **Payoffs:** Daily profits in UGX '000
- **Strategic Nature:** Classic Prisoner's Dilemma - mutual defection dominates cooperation

### 2. Promotion Day Game (Coordination Game)
- **Decision:** Each shop chooses which day to run their lunch promotion - Tuesday or Thursday
- **Market Dynamics:**
  - Tuesday is the busier day (higher demand)
  - If both promote on the same day, they split the crowd
  - If they promote on different days, each captures the full benefit of promoting on their chosen day
- **Payoffs:** Daily profits in UGX '000
- **Strategic Nature:** Coordination/Anti-coordination game with multiple pure Nash equilibria and a mixed-strategy equilibrium

**Research Questions:**
1. What are the equilibrium outcomes under simultaneous play?
2. Do dominant strategies exist? If so, what are they?
3. Are there pure or mixed strategy Nash equilibria?
4. How do outcomes change with sequential play (first-mover advantage)?
5. Can cooperation be sustained in repeated interactions?
6. How sensitive are results to changes in market conditions (costs, demand, competition)?

## Theoretical Background

This project implements the following game-theoretic concepts:

| Concept | Description | Implementation |
| --- | --- | --- |
| **Normal Form Games** | Simultaneous-move games represented by payoff matrices | `TwoPlayerGame` class with matrices A (row player) and B (column player) |
| **Best Response** | Strategy that maximizes a player's payoff given opponent's strategy | Computed with tie-breaking support (all strategies yielding max payoff) |
| **Dominant Strategy** | A strategy that yields higher (strict) or >= (weak) payoff than all others regardless of opponent | Implemented for both strict and weak dominance |
| **Iterated Elimination of Strictly Dominated Strategies (IESDS)** | Systematic removal of strictly dominated strategies until no more remain | Iterative algorithm with step-by-step tracking |
| **Pure Strategy Nash Equilibrium** | Strategy profile where no player can unilaterally improve by deviating | Identified as cells where each strategy is a best response to the other |
| **Mixed Strategy Nash Equilibrium** | Probability distributions over strategies where players are indifferent among played strategies | Support enumeration algorithm solving indifference conditions via linear systems |
| **Extensive Form Games** | Sequential-move games represented as game trees | Directed graphs using NetworkX with decision nodes and leaf payoffs |
| **Subgame Perfect Nash Equilibrium (SPNE)** | Equilibrium found by backward induction in sequential games | Solves tree from leaves to root, handles ties |
| **Repeated Games** | Games played multiple times with discounting | Simulates strategies, computes discount factor thresholds for cooperation |
| **Trigger Strategies** | Grim trigger and Tit-for-Tat for sustaining cooperation in Prisoner's Dilemmas | Implemented with theoretical threshold calculations |

## Features

### Core Analysis Modules

- **Game Representation (`game.py`)**: Object-oriented representation with input validation, JSON serialization/deserialization, interactive game creation, and formatted display using pandas DataFrames
- **Best Response Analysis (`analysis.py`)**: Computes best response correspondences for both players, generates best-response tables, and marks best responses on payoff matrices with asterisks
- **Equilibrium Analysis (`equilibrium.py`)**: Identifies strict and weak dominant strategies, finds dominated strategies, performs IESDS, and locates all pure-strategy Nash equilibria
- **Mixed Strategy Solver (`mixed.py`)**: Implements support enumeration over all possible support sizes, solves indifference equations, validates equilibrium conditions, provides 2x2 closed-form solution, and cross-checks with pure-strategy results
- **Game Tree Analysis (`gametree.py`)**: Constructs extensive-form trees, converts normal-form to sequential games, performs backward induction, handles indifference, visualizes trees with NetworkX/Matplotlib, and compares simultaneous vs sequential outcomes
- **Repeated Game Analysis (`repeated.py`)**: Implements standard strategies (Always Cooperate, Always Defect, Tit-for-Tat, Grim Trigger), runs round-robin tournaments, calculates cooperation thresholds for grim trigger and TFT, analyzes finite-horizon games via backward induction
- **Scenario Analysis (`scenarios.py`)**: Parameterized game builders, predefined scenarios, single-parameter sweeps, regime boundary detection, automatic game classification (Prisoner's Dilemma, Coordination, Anti-coordination, etc.), and CSV export
- **Visualization (`visualize.py`)**: Creates 4 publication-quality figures with consistent styling, color palettes, best-response diagrams showing mixed strategy equilibria, cooperation threshold curves, and Tuesday advantage sweeps

### Additional Features

- **Unified CLI Interface (`main.py`)**: Menu-driven interface providing access to all analysis features, interactive game creation, full analysis reports, and visualization generation
- **Comprehensive Validation**: Input validation, numerical tolerance handling (TOL=1e-9), finite value checks, and structural validation
- **Testing Framework**: 235 tests with property-based testing on random games, cross-validation against Nashpy, and hand-calculated verification
- **Export Capabilities**: JSON save/load for games, CSV exports for scenario tables and sweeps, PNG figures at 150 DPI

## Project Structure

```
game_theory_project/
├── main.py                 # Unified CLI interface
├── conftest.py             # Pytest configuration for module imports
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
│
├── src/                    # Core source modules
│   ├── __init__.py
│   ├── game.py             # Game representation and construction
│   ├── analysis.py         # Best response computation
│   ├── equilibrium.py      # Dominance and pure Nash equilibria
│   ├── mixed.py            # Mixed strategy Nash equilibria
│   ├── gametree.py         # Sequential games and backward induction
│   ├── repeated.py         # Repeated games and strategy simulations
│   ├── scenarios.py        # Scenario analysis and parameter sweeps
│   └── visualize.py        # Data visualization
│
├── tests/                  # Test suite (235 tests)
│   ├── pytest.ini
│   ├── test_stage3_4.py    # Best responses, dominance, pure Nash
│   ├── test_stage5.py      # Mixed strategies
│   ├── test_stage5_tree.py # Game trees
│   ├── test_stage5_repeated.py # Repeated games
│   ├── test_stage5_scenarios.py # Scenarios
│   └── test_stage5_visualize.py # Visualizations
│
├── docs/                   # Documentation and data files
│   ├── Game_Theory_Study_Guide.pdf
│   ├── scenarios_price_war.csv
│   ├── scenarios_promo_day.csv
│   └── *.png               # Example figures
│
├── figures/                # Generated visualizations
└── outputs/                # Generated analysis reports (CSV)
```

## Installation

### Prerequisites

- Python 3.8 or higher (tested on Python 3.13)
- pip package manager
- Virtual environment (recommended for dependency isolation)

### Setup Instructions

**On Windows:**

```bash
# Navigate to project directory
cd game_theory_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

**On Linux/macOS:**

```bash
# Navigate to project directory
cd game_theory_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Unified CLI Interface (Recommended)

The easiest way to use all features through an interactive menu:

```bash
# Run the main CLI application
python main.py
```

The menu provides:
1. Load Project Games (Price War or Promotion Day)
2. Create Custom Game (interactive input)
3. Display Current Game
4. Best Response Analysis
5. Dominance & Pure Nash Equilibria
6. Mixed Strategy Analysis
7. Sequential Game Analysis (Game Trees)
8. Repeated Game Analysis
9. Scenario Analysis
10. Generate All Visualizations
11. Run Full Analysis on Current Game
0. Exit

### Option 2: Graphical User Interface (GUI)

For a more user-friendly visual experience with buttons, tabs, and plots:

```bash
# Run the GUI application
python gui.py
```

The GUI provides:
- Menu-driven access to all analysis features
- Interactive game loading and creation
- Tabbed interface for game view and analysis results
- Built-in visualizations with matplotlib integration
- Scenario analysis tables
- Export games to JSON and load them back
- Generate all figures with one click

### Option 3: Run Individual Module Demonstrations

Each module has a `__main__` block that demonstrates its functionality:

```bash
# Game representation
python -m src.game

# Best response analysis
python -m src.analysis

# Dominance and pure Nash equilibria
python -m src.equilibrium

# Mixed strategy Nash equilibria
python -m src.mixed

# Sequential games (game trees)
python -m src.gametree

# Repeated games
python -m src.repeated

# Scenario analysis with CSV export
python -m src.scenarios

# Generate all visualizations
python -m src.visualize
```

Each module has a `__main__` block that demonstrates its functionality:

```bash
# Game representation
python -m src.game

# Best response analysis
python -m src.analysis

# Dominance and pure Nash equilibria
python -m src.equilibrium

# Mixed strategy Nash equilibria
python -m src.mixed

# Sequential games (game trees)
python -m src.gametree

# Repeated games
python -m src.repeated

# Scenario analysis with CSV export
python -m src.scenarios

# Generate all visualizations
python -m src.visualize
```


### Option 3: Programmatic API Usage

```python
from src.game import price_war_game, promo_day_game, TwoPlayerGame
from src.analysis import best_responses_row, best_responses_col, report as analysis_report
from src.equilibrium import pure_nash_equilibria, dominant_strategy, report as eq_report
from src.mixed import all_nash_equilibria, expected_payoffs, report as mixed_report
from src.gametree import GameTree, backward_induction
from src.repeated import grim_threshold, cooperation_sustainable

# Load and analyze a game
g = price_war_game()
g.display()

# Find Nash equilibria
eq_report(g)
mixed_report(g)

# Analyze as sequential game
tree = GameTree.from_normal_form(g)
result = backward_induction(tree)
print(f"SPNE payoffs: {result['payoffs']}")

# Check cooperation threshold in repeated game
delta_star = grim_threshold(g)
print(f"Minimum discount factor for cooperation: {delta_star:.4f}")
```

### Option 4: Create and Analyze Custom Games

```python
from src.game import TwoPlayerGame
import numpy as np

# Define Battle of the Sexes
g = TwoPlayerGame.from_cells(
    name="Battle of the Sexes",
    players=["Husband", "Wife"],
    row_strategies=["Opera", "Football"],
    col_strategies=["Opera", "Football"],
    cells=[
        [(2, 1), (0, 0)],
        [(0, 0), (1, 2)]
    ]
)
g.display()
```

Or use the interactive builder:
```python
from src.game import game_from_user
g = game_from_user()  # Prompts for players, strategies, payoffs
```

### Option 5: Run Tests

```bash
# Run all tests
python -m pytest -v

# Run with more detailed output
python -m pytest -v --tb=short

# Run specific test file
python -m pytest tests/test_stage3_4.py -v

# Run tests matching a pattern
python -m pytest -k "nash" -v
```

## Mathematical Formulations

### Best Response

For row player choosing strategy $i$ against column player's mixed strategy $q \in \Delta S_2$:
$$
BR_1(q) = \arg\max_{i \in S_1} \sum_{j \in S_2} A_{ij} q_j
$$

### Nash Equilibrium (Pure)

A strategy profile $(i^*, j^*)$ is a pure Nash equilibrium if:
$$
A_{i^*, j^*} \ge A_{i, j^*} \ \forall i, \qquad B_{i^*, j^*} \ge B_{i^*, j} \ \forall j
$$

### Mixed Strategy Nash Equilibrium

For a 2×2 game, let $p$ be probability row plays strategy 1, $q$ probability column plays strategy 1. Row is indifferent when:
$$
q = \frac{A_{21} - A_{11}}{A_{11} - A_{12} - A_{21} + A_{22}}, \qquad p = \frac{B_{22} - B_{21}}{B_{11} - B_{12} - B_{21} + B_{22}}
$$

For larger games, the system solves:
$$
A_{I,J}\, q_J = v \cdot \mathbf{1}, \quad \sum_{j \in J} q_j = 1, \qquad p_I\, B_{I,J} = w \cdot \mathbf{1}, \quad \sum_{i \in I} p_i = 1
$$
subject to $p>0, q>0$ and no profitable deviations outside supports.

### Grim Trigger (Infinitely Repeated Prisoner's Dilemma)

Cooperation is sustainable as a subgame perfect equilibrium when:
$$
\delta \ge \delta^* = \frac{T - R}{T - P}
$$

Where: $T$ = Temptation (defect when cooperate), $R$ = Reward (mutual cooperate), $P$ = Punishment (mutual defect), $S$ = Sucker (cooperate when defect)

Payoff from cooperation forever: $V_C = \frac{R}{1-\delta}$  
Payoff from one-shot defection: $V_D = T + \frac{\delta P}{1-\delta}$

### Tit-for-Tat Thresholds

TFT sustains cooperation when $\delta \ge \max\{\frac{T-R}{R-S}, \frac{T-R}{T-P}\}$

## Results and Analysis

### Price War Game (Prisoner's Dilemma)

| Analysis | Result |
| --- | --- |
| Payoff Structure | T=120, R=100, P=75, S=40 |
| Dominant Strategy | Cut for both players (strict) |
| Pure Nash Equilibrium | (Cut, Cut) - unique |
| Mixed Strategy Equilibrium | None exists |
| Grim Trigger Threshold | $\delta^* = (120-100)/(120-75) = 20/45 \approx 0.444$ |
| Tit-for-Tat | Requires same threshold in this symmetric case |

**Interpretation:** The Nash equilibrium is inefficient (Pareto-dominated by mutual cooperation at Keep/Keep with payoffs 100,100). However, in infinitely repeated interactions with sufficient patience ($\delta \ge 0.444$), cooperation can be sustained through credible punishment strategies.

### Promotion Day Game (Anti-coordination)

| Analysis | Result |
| --- | --- |
| Type | Anti-coordination game (similar to Battle of the Sexes) |
| Dominant Strategy | None |
| Pure Nash Equilibria | (Tue, Thu) and (Thu, Tue) |
| Mixed Strategy Equilibrium | $p=q=(0.75, 0.25)$ - each plays Tuesday with 75% probability |
| Expected Payoffs (Mixed) | (42.5, 42.5) for both players |
| Sequential Play (Shop A first) | Shop A moves first, leads to (Tue, Thu) with payoffs (80, 50) - first-mover advantage |

**Interpretation:** Coordination on different days is preferred. When mixed strategies are used, there's a 56.25% chance of collision (both on Tuesday) and 6.25% chance both on Thursday, with expected payoffs below the pure Nash payoffs. First-mover advantage exists in sequential play.

### Scenario Analysis Insights

- **Price War Sensitivity:** As costs increase beyond certain thresholds, the game transitions from Prisoner's Dilemma to coordination game to games where keeping prices dominates (no price war incentive)
- **Promotion Dynamics:** Increasing Tuesday's advantage makes it more likely to be chosen, eventually becoming a dominant strategy and eliminating the mixed strategy equilibrium
- **Cooperation Sustainability:** Lower costs make price cutting more tempting (higher $\delta^*$ required for cooperation), while higher costs make cooperation easier to sustain

## Testing and Validation

### Test Coverage

The project includes **235 tests** across 6 test modules, all passing:

| Test Module | Focus | Tests |
| --- | --- | --- |
| `test_stage3_4.py` | Best responses, dominance, IESDS, pure Nash | ~30 tests |
| `test_stage5.py` | Mixed strategies, support enumeration | ~20 tests |
| `test_stage5_tree.py` | Game trees, backward induction | ~8 tests |
| `test_stage5_repeated.py` | Repeated games, strategies, thresholds | ~50 tests |
| `test_stage5_scenarios.py` | Scenarios, sweeps, regime classification | ~50 tests |
| `test_stage5_visualize.py` | Visualization functions | ~77 tests |

### Validation Methods

1. **Hand Calculations**: All textbook examples verified manually (Prisoner's Dilemma, Battle of the Sexes, Matching Pennies, Stag Hunt, Rock-Paper-Scissors)
2. **Cross-Validation**: Results compared against Nashpy library on 40-60 random games per test with seeds for reproducibility
3. **Property Testing**: Verified that all computed equilibria satisfy Nash equilibrium conditions ($\max_i A_i q \le u_1 + \epsilon$, $\max_j p B_j \le u_2 + \epsilon$)
4. **Structural Validation**: Tested game construction, input validation, and edge cases
5. **Numerical Precision**: Used appropriate tolerances (TOL=1e-9) for floating-point comparisons

### Running Validation

```bash
# Run all tests with verbose output
python -m pytest -v

# Expected output: 235 passed in ~30-60s
```

## Dependencies

| Library | Version | Purpose |
| --- | --- | --- |
| `numpy` | >=1.20.0 | Numerical computations, matrix operations, linear algebra for mixed strategy solving |
| `pandas` | >=1.3.0 | Tabular data structures, DataFrame display of payoff matrices |
| `matplotlib` | >=3.4.0 | Figure generation, plotting best-response diagrams and sweep curves |
| `networkx` | >=2.6.0 | Game tree representation, traversal, and visualization |
| `nashpy` | >=0.0.35 | Independent validation of Nash equilibrium calculations |
| `jupyter` | >=1.0.0 | Notebook interface support (optional) |
| `pytest` | >=7.0.0 | Testing framework |

All dependencies are listed in `requirements.txt`.

## Limitations

- **Game Size**: Mixed strategy solver uses support enumeration which is exponential in strategy count; efficient enough for games up to ~4×4 in practice, tested up to reasonable sizes
- **Degenerate Games**: Singular linear systems in support enumeration are skipped (with note in code) as per standard approach for non-degenerate games
- **Game Types**: Focused on two-player finite games; doesn't support n-player games, games with incomplete information, or Bayesian games
- **Cooperative Games**: While repeated games are covered, core cooperative game theory concepts (coalition games, Shapley value, core, nucleolus) are not implemented
- **UI**: Command-line interface only; no graphical user interface (GUI) or web interface
- **Perfect Information**: Game trees assume perfect information; extensive form games with imperfect information not fully modeled

## Future Improvements

- [ ] Support for n-player games (3+ players)
- [ ] Implementation of cooperative game theory solutions (Shapley value, core)
- [ ] Bayesian games and games with incomplete information
- [ ] More sophisticated mixed strategy solvers for larger games (Lemke-Howson algorithm)
- [ ] Web-based interface using Flask/Streamlit
- [ ] Export to LaTeX for report generation
- [ ] Additional solution concepts (evolutionarily stable strategies, trembling hand perfect equilibrium)
- [ ] Support for continuous strategy spaces

## License

This project is developed for academic purposes as part of a Game Theory course. All code is provided for educational use.

## Acknowledgments

- **Nashpy** - Used for independent validation of Nash equilibrium computations
- **NumPy, pandas, matplotlib, NetworkX** - Core scientific Python stack
- **Game Theory Literature** - Implementations follow standard algorithms from Osborne & Rubinstein, Fudenberg & Tirole, and other standard texts
- **Course Material** - Based on the semester rolling project requirements for Game Theory course
- **Testing Approach** - Cross-validation methodology ensures correctness of core algorithms implemented from scratch
