#!/usr/bin/env python3
"""
Game Theory Analysis System - GUI Interface
A graphical user interface for analyzing strategic interactions.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.game import TwoPlayerGame, price_war_game, promo_day_game, game_from_user
from src.analysis import best_responses_row, best_responses_col, report as analysis_report
from src.equilibrium import pure_nash_equilibria, dominant_strategy, dominated_strategies, report as equilibrium_report
from src.mixed import all_nash_equilibria, mixed_equilibrium_2x2, expected_payoffs, is_pure, report as mixed_report
from src.gametree import GameTree, backward_induction, report as tree_report
from src.repeated import report as repeated_report, grim_threshold, pd_parameters
from src.scenarios import build_price_war, build_promo_day, scenario_table, PRICE_WAR_SCENARIOS, PROMO_SCENARIOS
from src.visualize import plot_payoff_matrices, plot_best_responses, plot_cooperation_threshold, plot_tuesday_sweep


class GameTheoryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Game Theory Analysis System")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        self.current_game = None
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create menu bar
        self.create_menu()
        
        # Create main container
        self.create_main_layout()
        
        # Status bar
        self.status_bar = ttk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Price War Game", command=self.load_price_war)
        file_menu.add_command(label="Load Promotion Day Game", command=self.load_promo_day)
        file_menu.add_command(label="Create Custom Game", command=self.create_custom_game)
        file_menu.add_separator()
        file_menu.add_command(label="Save Game as JSON", command=self.save_game)
        file_menu.add_command(label="Load Game from JSON", command=self.load_game_json)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Best Response Analysis", command=self.best_response_analysis)
        analysis_menu.add_command(label="Dominance & Pure Nash", command=self.dominance_analysis)
        analysis_menu.add_command(label="Mixed Strategy Analysis", command=self.mixed_analysis)
        analysis_menu.add_command(label="Sequential Game (Game Tree)", command=self.sequential_analysis)
        analysis_menu.add_command(label="Repeated Game Analysis", command=self.repeated_analysis)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Full Analysis", command=self.full_analysis)
        
        # Visualize menu
        viz_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualize", menu=viz_menu)
        viz_menu.add_command(label="Show Payoff Matrix", command=self.show_payoff_matrix)
        viz_menu.add_command(label="Best Response Diagram", command=self.show_best_response_diagram)
        viz_menu.add_command(label="Game Tree (Sequential)", command=self.show_game_tree)
        viz_menu.add_separator()
        viz_menu.add_command(label="Generate All Figures", command=self.generate_all_figures)
        
        # Scenario menu
        scenario_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Scenarios", menu=scenario_menu)
        scenario_menu.add_command(label="Price War Scenarios", command=self.price_war_scenarios)
        scenario_menu.add_command(label="Promotion Day Scenarios", command=self.promo_scenarios)
        scenario_menu.add_separator()
        scenario_menu.add_command(label="Run Full Scenario Analysis", command=self.run_scenario_analysis)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def create_main_layout(self):
        # Top frame - Game selection
        top_frame = ttk.LabelFrame(self.root, text="Current Game", padding=10)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Load Price War", command=self.load_price_war).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Load Promotion Day", command=self.load_promo_day).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Create Custom Game", command=self.create_custom_game).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Load from JSON", command=self.load_game_json).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Save as JSON", command=self.save_game).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear", command=self.clear_game).pack(side=tk.LEFT, padx=2)
        
        self.game_label = ttk.Label(top_frame, text="No game loaded", font=('Arial', 9))
        self.game_label.pack(anchor=tk.W, pady=5)
        
        # Main content area with notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tab 1: Game Display
        self.game_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.game_tab, text="Game View")
        
        # Tab 2: Analysis Results
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="Analysis")
        
        self.analysis_text = scrolledtext.ScrolledText(self.analysis_tab, wrap=tk.WORD, font=('Courier', 9))
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: Visualization
        self.viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="Visualization")
        
        self.viz_frame = ttk.Frame(self.viz_tab)
        self.viz_frame.pack(fill=tk.BOTH, expand=True)
        
    def update_status(self, message):
        self.status_bar.config(text=message)
        self.root.update_idletasks()
        
    def load_price_war(self):
        try:
            self.current_game = price_war_game()
            self.update_display()
            self.update_status("Loaded: Lunch Price War Game")
            messagebox.showinfo("Success", "Price War Game loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load game: {e}")
            
    def load_promo_day(self):
        try:
            self.current_game = promo_day_game()
            self.update_display()
            self.update_status("Loaded: Promotion Day Game")
            messagebox.showinfo("Success", "Promotion Day Game loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load game: {e}")
            
    def create_custom_game(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Create Custom Game")
        dialog.geometry("550x520")
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Game info
        info_frame = ttk.LabelFrame(dialog, text="Game Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(info_frame, text="Game Name:").grid(row=0, column=0, sticky=tk.W)
        game_name = ttk.Entry(info_frame, width=40)
        game_name.insert(0, "Custom Game")
        game_name.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(info_frame, text="Player 1 (Rows):").grid(row=1, column=0, sticky=tk.W)
        p1_name = ttk.Entry(info_frame, width=40)
        p1_name.insert(0, "Player 1")
        p1_name.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(info_frame, text="Player 2 (Columns):").grid(row=2, column=0, sticky=tk.W)
        p2_name = ttk.Entry(info_frame, width=40)
        p2_name.insert(0, "Player 2")
        p2_name.grid(row=2, column=1, padx=5, pady=2)
        
        # Strategies
        strat_frame = ttk.LabelFrame(dialog, text="Strategies", padding=10)
        strat_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(strat_frame, text="Strategies for Player 1 (comma separated):").grid(row=0, column=0, sticky=tk.W)
        p1_strats = ttk.Entry(strat_frame, width=40)
        p1_strats.insert(0, "A,B")
        p1_strats.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(strat_frame, text="Strategies for Player 2 (comma separated):").grid(row=1, column=0, sticky=tk.W)
        p2_strats = ttk.Entry(strat_frame, width=40)
        p2_strats.insert(0, "A,B")
        p2_strats.grid(row=1, column=1, padx=5, pady=2)
        
        # Payoffs
        payoff_frame = ttk.LabelFrame(dialog, text="Payoffs - Enter as (p1,p2)", padding=10)
        payoff_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Label(payoff_frame, text="Format: One row per line, cells separated by commas").pack(anchor=tk.W)
        ttk.Label(payoff_frame, text="Example for 2x2: (1,2),(3,4) and (5,6),(7,8)").pack(anchor=tk.W)
        
        payoff_text = scrolledtext.ScrolledText(payoff_frame, height=6)
        payoff_text.pack(fill=tk.BOTH, expand=True, pady=5)
        payoff_text.insert('1.0', "(1,2),(3,4)\n(5,6),(7,8)")
        
        def create_game():
            try:
                name = game_name.get().strip() or "Custom Game"
                p1 = p1_name.get().strip() or "Player 1"
                p2 = p2_name.get().strip() or "Player 2"
                if p2 == p1:
                    p2 = f"{p2} (2)"
                
                rows = [s.strip() for s in p1_strats.get().split(',')]
                cols = [s.strip() for s in p2_strats.get().split(',')]
                
                # Parse payoffs
                payoff_str = payoff_text.get('1.0', tk.END)
                cells = []
                for line in payoff_str.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('Format') or line.startswith('Example'):
                        continue
                    # Parse (p1,p2),(p1,p2)
                    import re
                    matches = re.findall(r'\(([^,]+),([^)]+)\)', line)
                    if matches:
                        row_cells = [(float(a.strip()), float(b.strip())) for a, b in matches]
                        cells.append(row_cells)
                
                if len(cells) != len(rows) or (cells and len(cells[0]) != len(cols)):
                    messagebox.showerror("Error", "Payoff dimensions don't match strategies", parent=dialog)
                    return
                
                from src.game import TwoPlayerGame
                self.current_game = TwoPlayerGame.from_cells(name, [p1, p2], rows, cols, cells)
                
                # Ask to save
                save = messagebox.askyesno("Save Game", "Save this game as JSON?", parent=dialog)
                if save:
                    filename = filedialog.asksaveasfilename(
                        parent=dialog,
                        defaultextension=".json",
                        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                        initialfile=f"{self.current_game.name.replace(' ', '_')}.json"
                    )
                    if filename:
                        self.current_game.save_json(filename)
                        messagebox.showinfo("Saved", f"Game saved to {filename}", parent=dialog)
                
                self.update_display()
                self.update_status(f"Loaded: {self.current_game.name}")
                dialog.destroy()
                messagebox.showinfo("Success", "Custom game created successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create game: {e}", parent=dialog)
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Create & Save Game", command=create_game).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
            
    def clear_game(self):
        self.current_game = None
        self.game_label.config(text="No game loaded")
        for widget in self.game_tab.winfo_children():
            widget.destroy()
        self.analysis_text.delete('1.0', tk.END)
        self.update_status("Game cleared")
        
    def update_display(self):
        if self.current_game is None:
            return
        self.game_label.config(text=f"Game: {self.current_game.name} | Players: {', '.join(self.current_game.players)}")
        # Clear and display in table format
        for widget in self.game_tab.winfo_children():
            widget.destroy()
        
        # Create frame for table
        table_frame = ttk.Frame(self.game_tab)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for payoff table
        cols = list(self.current_game.col_strategies)
        tree = ttk.Treeview(table_frame, columns=cols, show='tree headings', height=min(10, len(self.current_game.row_strategies)+2))
        tree.heading('#0', text=f"{self.current_game.players[0]} \\ {self.current_game.players[1]}")
        tree.column('#0', width=150, minwidth=100)
        
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, minwidth=100, anchor=tk.CENTER)
        
        # Insert data
        for i, rstrat in enumerate(self.current_game.row_strategies):
            values = []
            for j in range(len(self.current_game.col_strategies)):
                values.append(f"({int(self.current_game.A[i,j]) if float(self.current_game.A[i,j]).is_integer() else self.current_game.A[i,j]}, {int(self.current_game.B[i,j]) if float(self.current_game.B[i,j]).is_integer() else self.current_game.B[i,j]})")
            tree.insert('', tk.END, text=rstrat, values=values)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.notebook.select(0)
        
    def save_game(self):
        if self.current_game is None:
            messagebox.showwarning("Warning", "No game loaded")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"{self.current_game.name.replace(' ', '_')}.json"
        )
        if filename:
            try:
                self.current_game.save_json(filename)
                messagebox.showinfo("Success", f"Game saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
                
    def load_game_json(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.current_game = TwoPlayerGame.load_json(filename)
                self.update_display()
                messagebox.showinfo("Success", "Game loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")
                
    def run_analysis(self, report_func):
        if self.current_game is None:
            messagebox.showwarning("Warning", "No game loaded")
            return
        try:
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                report_func(self.current_game)
            self.analysis_text.delete('1.0', tk.END)
            self.analysis_text.insert('1.0', f.getvalue())
            self.notebook.select(1)
            self.update_status("Analysis complete")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            
    def best_response_analysis(self):
        self.run_analysis(analysis_report)
        
    def dominance_analysis(self):
        self.run_analysis(equilibrium_report)
        
    def mixed_analysis(self):
        self.run_analysis(mixed_report)
        
    def sequential_analysis(self):
        if self.current_game is None:
            messagebox.showwarning("Warning", "No game loaded")
            return
        try:
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                tree = GameTree.from_normal_form(self.current_game)
                tree_report(tree)
                from src.gametree import compare_with_simultaneous
                compare_with_simultaneous(self.current_game)
            self.analysis_text.delete('1.0', tk.END)
            self.analysis_text.insert('1.0', f.getvalue())
            self.notebook.select(1)
            self.update_status("Sequential analysis complete")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            
    def repeated_analysis(self):
        self.run_analysis(repeated_report)
        
    def full_analysis(self):
        if self.current_game is None:
            messagebox.showwarning("Warning", "No game loaded")
            return
        try:
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                print("="*60)
                print(f"FULL ANALYSIS: {self.current_game.name}")
                print("="*60)
                print("\n### 1. GAME REPRESENTATION ###")
                self.current_game.display()
                print("\n### 2. BEST RESPONSE ANALYSIS ###")
                analysis_report(self.current_game)
                print("\n### 3. DOMINANCE & PURE NASH EQUILIBRIA ###")
                equilibrium_report(self.current_game)
                print("\n### 4. MIXED STRATEGY ANALYSIS ###")
                mixed_report(self.current_game)
                print("\n### 5. SEQUENTIAL GAME ANALYSIS ###")
                tree = GameTree.from_normal_form(self.current_game)
                tree_report(tree)
                from src.gametree import compare_with_simultaneous
                compare_with_simultaneous(self.current_game)
                print("\n### 6. REPEATED GAME ANALYSIS ###")
                repeated_report(self.current_game)
                print("\n" + "="*60)
            self.analysis_text.delete('1.0', tk.END)
            self.analysis_text.insert('1.0', f.getvalue())
            self.notebook.select(1)
            self.update_status("Full analysis complete")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            
    def show_payoff_matrix(self):
        if self.current_game is None:
            messagebox.showwarning("Warning", "No game loaded")
            return
        try:
            for widget in self.viz_frame.winfo_children():
                widget.destroy()
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            df = self.current_game.to_dataframe()
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, 
                           loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax.set_title(f"{self.current_game.name} - Payoff Matrix", pad=20)
            canvas = FigureCanvasTkAgg(fig, self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.notebook.select(2)
            self.update_status("Payoff matrix displayed")
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {e}")
            
    def show_best_response_diagram(self):
        if self.current_game is None:
            messagebox.showwarning("Warning", "No game loaded")
            return
        try:
            for widget in self.viz_frame.winfo_children():
                widget.destroy()
            fig, ax = plt.subplots(figsize=(6, 5))
            p1, p2 = self.current_game.players
            from src.visualize import _best_response_path, _equilibrium_points, _style
            from matplotlib.ticker import FuncFormatter
            
            _style(ax)
            row_path = _best_response_path(self.current_game.A)
            col_path = _best_response_path(self.current_game.B.T)
            if row_path:
                xs, ys = zip(*row_path)
                ax.plot(xs, ys, color='#2a78d6', linewidth=2.2, label=f"{p1}'s BR")
            if col_path:
                ys, xs = zip(*col_path)
                ax.plot(xs, ys, color='#eb6834', linewidth=2.2, label=f"{p2}'s BR")
            for x, y, label in _equilibrium_points(self.current_game):
                ax.plot([x], [y], marker="o", markersize=9, markerfacecolor='black')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_xlabel(f"P({p2} plays {self.current_game.col_strategies[0]})")
            ax.set_ylabel(f"P({p1} plays {self.current_game.row_strategies[0]})")
            ax.legend()
            ax.set_title(f"Best Response Diagram - {self.current_game.name}")
            canvas = FigureCanvasTkAgg(fig, self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.notebook.select(2)
            self.update_status("Best response diagram displayed")
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {e}")
            
    def show_game_tree(self):
        if self.current_game is None:
            messagebox.showwarning("Warning", "No game loaded")
            return
        try:
            for widget in self.viz_frame.winfo_children():
                widget.destroy()
            from src.gametree import GameTree, backward_induction, draw
            import tempfile
            tree = GameTree.from_normal_form(self.current_game)
            backward_induction(tree)
            tmpfile = tempfile.mktemp(suffix='.png')
            draw(tree, tmpfile)
            try:
                from PIL import Image, ImageTk
                img = Image.open(tmpfile)
                img = img.resize((700, 500), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                label = ttk.Label(self.viz_frame, image=photo)
                label.image = photo
                label.pack(fill=tk.BOTH, expand=True)
            except Exception:
                from matplotlib.image import imread
                fig, ax = plt.subplots(figsize=(7, 5))
                img = imread(tmpfile)
                ax.imshow(img)
                ax.axis('off')
                canvas = FigureCanvasTkAgg(fig, self.viz_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.notebook.select(2)
            self.update_status("Game tree displayed")
            try:
                os.unlink(tmpfile)
            except:
                pass
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {e}")
            
    def generate_all_figures(self):
        try:
            os.makedirs('figures', exist_ok=True)
            from src.visualize import make_all
            info = make_all('figures')
            messagebox.showinfo("Success", 
                f"Generated 4 figures in figures/\n\n"
                f"Cost boundaries: {info['cost_boundaries'][0]:,.0f}, {info['cost_boundaries'][1]:,.0f} UGX\n"
                f"Tuesday boundary: {info['tuesday_boundary']:.1f} UGX")
            self.update_status("All figures generated")
        except Exception as e:
            messagebox.showerror("Error", f"Generation failed: {e}")
            
    def price_war_scenarios(self):
        try:
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                from src.scenarios import scenario_table, build_price_war, PRICE_WAR_SCENARIOS
                table = scenario_table(build_price_war, PRICE_WAR_SCENARIOS)
                print(table.to_string(index=False))
            self.analysis_text.delete('1.0', tk.END)
            self.analysis_text.insert('1.0', f.getvalue())
            self.notebook.select(1)
            self.update_status("Price war scenarios displayed")
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")
            
    def promo_scenarios(self):
        try:
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                from src.scenarios import scenario_table, build_promo_day, PROMO_SCENARIOS
                table = scenario_table(build_promo_day, PROMO_SCENARIOS)
                print(table.to_string(index=False))
            self.analysis_text.delete('1.0', tk.END)
            self.analysis_text.insert('1.0', f.getvalue())
            self.notebook.select(1)
            self.update_status("Promotion scenarios displayed")
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")
            
    def run_scenario_analysis(self):
        try:
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                from src.scenarios import report
                report()
            self.analysis_text.delete('1.0', tk.END)
            self.analysis_text.insert('1.0', f.getvalue())
            self.notebook.select(1)
            self.update_status("Scenario analysis complete - check outputs/")
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")
            
    def show_about(self):
        about_text = """Game Theory Analysis System

Version: 1.0
Course: Game Theory
Project: Design and Implementation of a Game-Theoretic Analysis System Using Python

Features:
• Normal form game representation and analysis
• Best response, dominance, and Nash equilibrium computation
• Mixed strategy analysis with support enumeration
• Sequential games and backward induction
• Repeated games with grim trigger and tit-for-tat
• Scenario analysis and parameter sweeps
• Publication-quality visualizations
• 235 comprehensive tests

Developed for academic purposes."""
        messagebox.showinfo("About", about_text)


def main():
    root = tk.Tk()
    app = GameTheoryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
