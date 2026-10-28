#!/usr/bin/env python3
"""
Game Theory Analysis System - Unified CLI
A command-line interface for analyzing strategic interactions between decision-makers.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.game import TwoPlayerGame, price_war_game, promo_day_game, game_from_user
from src.analysis import report as analysis_report
from src.equilibrium import report as equilibrium_report
from src.mixed import report as mixed_report
from src.gametree import GameTree, report as tree_report, compare_with_simultaneous, draw
from src.repeated import report as repeated_report
from src.scenarios import report as scenarios_report, build_price_war, build_promo_day
from src.visualize import make_all


class GameTheoryCLI:
    def __init__(self):
        self.current_game = None
        
    def display_menu(self):
        print("\n" + "="*60)
        print("GAME THEORY ANALYSIS SYSTEM")
        print("="*60)
        print("\n1. Load Project Games")
        print("2. Create Custom Game")
        print("3. Display Current Game")
        print("4. Best Response Analysis")
        print("5. Dominance & Pure Nash Equilibria")
        print("6. Mixed Strategy Analysis")
        print("7. Sequential Game Analysis (Game Trees)")
        print("8. Repeated Game Analysis")
        print("9. Scenario Analysis")
        print("10. Generate All Visualizations")
        print("11. Run Full Analysis on Current Game")
        print("0. Exit")
        print("\n" + "="*60)
        
    def load_project_games(self):
        print("\n" + "-"*60)
        print("PROJECT GAMES")
        print("-"*60)
        print("1. Lunch Price War")
        print("2. Promotion Day Game")
        print("3. Both Games")
        print("0. Back to Main Menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            self.current_game = price_war_game()
            print("\n✓ Loaded: Lunch Price War")
            self.current_game.display()
        elif choice == '2':
            self.current_game = promo_day_game()
            print("\n✓ Loaded: Promotion Day Game")
            self.current_game.display()
        elif choice == '3':
            print("\n=== LUNCH PRICE WAR ===")
            g1 = price_war_game()
            g1.display()
            print("\n=== PROMOTION DAY GAME ===")
            g2 = promo_day_game()
            g2.display()
            input("\nPress Enter to continue...")
        elif choice == '0':
            return
        else:
            print("\n✗ Invalid option")
            input("\nPress Enter to continue...")
            
    def create_custom_game(self):
        print("\n" + "-"*60)
        print("CREATE CUSTOM GAME")
        print("-"*60)
        try:
            self.current_game = game_from_user()
            print("\n✓ Game created successfully!")
            self.current_game.display()
        except Exception as e:
            print(f"\n✗ Error: {e}")
        input("\nPress Enter to continue...")
        
    def display_current_game(self):
        if self.current_game is None:
            print("\n✗ No game loaded. Please load or create a game first.")
        else:
            self.current_game.display()
        input("\nPress Enter to continue...")
        
    def best_response_analysis(self):
        if self.current_game is None:
            print("\n✗ No game loaded. Please load or create a game first.")
            input("\nPress Enter to continue...")
            return
        try:
            analysis_report(self.current_game)
        except Exception as e:
            print(f"\n✗ Error: {e}")
        input("\nPress Enter to continue...")
        
    def dominance_nash_analysis(self):
        if self.current_game is None:
            print("\n✗ No game loaded. Please load or create a game first.")
            input("\nPress Enter to continue...")
            return
        try:
            equilibrium_report(self.current_game)
        except Exception as e:
            print(f"\n✗ Error: {e}")
        input("\nPress Enter to continue...")
        
    def mixed_strategy_analysis(self):
        if self.current_game is None:
            print("\n✗ No game loaded. Please load or create a game first.")
            input("\nPress Enter to continue...")
            return
        try:
            mixed_report(self.current_game)
        except Exception as e:
            print(f"\n✗ Error: {e}")
        input("\nPress Enter to continue...")
        
    def sequential_game_analysis(self):
        if self.current_game is None:
            print("\n✗ No game loaded. Please load or create a game first.")
            input("\nPress Enter to continue...")
            return
        try:
            tree = GameTree.from_normal_form(self.current_game)
            tree_report(tree)
            compare_with_simultaneous(self.current_game)
            
            save_fig = input("\nSave game tree visualization? (y/n): ").strip().lower()
            if save_fig == 'y':
                os.makedirs('figures', exist_ok=True)
                slug = self.current_game.name.lower().replace(" ", "_")
                filename = f"figures/tree_{slug}_custom.png"
                draw(tree, filename)
                print(f"✓ Saved to {filename}")
        except Exception as e:
            print(f"\n✗ Error: {e}")
        input("\nPress Enter to continue...")
        
    def repeated_game_analysis(self):
        if self.current_game is None:
            print("\n✗ No game loaded. Please load or create a game first.")
            input("\nPress Enter to continue...")
            return
        try:
            repeated_report(self.current_game)
        except Exception as e:
            print(f"\n✗ Error: {e}")
        input("\nPress Enter to continue...")
        
    def scenario_analysis(self):
        print("\n" + "-"*60)
        print("SCENARIO ANALYSIS")
        print("-"*60)
        print("This will run scenario analysis on both project games,")
        print("generate CSV reports in outputs/ directory.")
        print("-"*60)
        
        confirm = input("\nRun scenario analysis? (y/n): ").strip().lower()
        if confirm == 'y':
            try:
                scenarios_report()
                print("\n✓ Scenario analysis complete!")
                print("  Reports saved to outputs/ directory")
            except Exception as e:
                print(f"\n✗ Error: {e}")
        input("\nPress Enter to continue...")
        
    def generate_visualizations(self):
        print("\n" + "-"*60)
        print("GENERATE VISUALIZATIONS")
        print("-"*60)
        print("This will generate all 4 figures for the project games:")
        print("  1. Payoff matrices with best responses and Nash equilibria")
        print("  2. Best-response diagrams")
        print("  3. Cooperation threshold vs cost")
        print("  4. Tuesday's advantage sweep")
        print("-"*60)
        
        confirm = input("\nGenerate all visualizations? (y/n): ").strip().lower()
        if confirm == 'y':
            try:
                os.makedirs('figures', exist_ok=True)
                info = make_all('figures')
                print("\n✓ Visualizations generated successfully!")
                print("  Saved to figures/ directory")
                print(f"\nRegime changes:")
                print(f"  Cost boundaries: {info['cost_boundaries'][0]:,.0f} and {info['cost_boundaries'][1]:,.0f} UGX")
                print(f"  Tuesday profit boundary: {info['tuesday_boundary']:.1f} UGX")
            except Exception as e:
                print(f"\n✗ Error: {e}")
        input("\nPress Enter to continue...")
        
    def full_analysis(self):
        if self.current_game is None:
            print("\n✗ No game loaded. Please load or create a game first.")
            input("\nPress Enter to continue...")
            return
        
        try:
            print("\n" + "="*60)
            print(f"FULL ANALYSIS: {self.current_game.name}")
            print("="*60)
            
            print("\n\n### 1. GAME REPRESENTATION ###")
            self.current_game.display()
            
            print("\n\n### 2. BEST RESPONSE ANALYSIS ###")
            analysis_report(self.current_game)
            
            print("\n\n### 3. DOMINANCE & PURE NASH EQUILIBRIA ###")
            equilibrium_report(self.current_game)
            
            print("\n\n### 4. MIXED STRATEGY ANALYSIS ###")
            mixed_report(self.current_game)
            
            print("\n\n### 5. SEQUENTIAL GAME ANALYSIS ###")
            tree = GameTree.from_normal_form(self.current_game)
            tree_report(tree)
            compare_with_simultaneous(self.current_game)
            
            print("\n\n### 6. REPEATED GAME ANALYSIS ###")
            repeated_report(self.current_game)
            
            print("\n" + "="*60)
            print("FULL ANALYSIS COMPLETE")
            print("="*60)
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
        input("\nPress Enter to continue...")
        
    def run(self):
        while True:
            self.display_menu()
            choice = input("\nEnter your choice: ").strip()
            
            if choice == '1':
                self.load_project_games()
            elif choice == '2':
                self.create_custom_game()
            elif choice == '3':
                self.display_current_game()
            elif choice == '4':
                self.best_response_analysis()
            elif choice == '5':
                self.dominance_nash_analysis()
            elif choice == '6':
                self.mixed_strategy_analysis()
            elif choice == '7':
                self.sequential_game_analysis()
            elif choice == '8':
                self.repeated_game_analysis()
            elif choice == '9':
                self.scenario_analysis()
            elif choice == '10':
                self.generate_visualizations()
            elif choice == '11':
                self.full_analysis()
            elif choice == '0':
                print("\nThank you for using Game Theory Analysis System!")
                break
            else:
                print("\n✗ Invalid option. Please try again.")
                input("\nPress Enter to continue...")


if __name__ == "__main__":
    cli = GameTheoryCLI()
    cli.run()
