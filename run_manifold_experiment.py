import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pyspiel
import copy
from typing import Dict, Any

from games.mnk_game import MNKGame
from policies.policy_registry import instantiate_policy
from policies.mcts.policy import MCTSAgent
from policies.intuitivegamer_depth_limited.policy import DepthLimitedIGPolicy

# --- 1. Robust Game Loader (Fixes the C++ Init Error) ---
def load_game_safe(game_config: Dict[str, Any]) -> Any:
    game_name = game_config["name"]
    game_params = game_config.get("parameters", {})
    
    if game_name == "mnk_game":
        # Direct instantiation of Python class (Bypasses C++ binding issues)
        return MNKGame(**game_params)
    
    # Fallback for standard OpenSpiel games
    return pyspiel.load_game(game_name, game_params)

# --- 2. Simulation Logic ---
def play_match(game, bot1, bot2):
    """Plays a single match, returns 1 if bot1 wins, 0 if bot2 wins, 0.5 for draw."""
    state = game.new_initial_state()
    bots = [bot1, bot2]
    
    while not state.is_terminal():
        current_player = state.current_player()
        action = bots[current_player].step(state)
        state.apply_action(action)
    
    # Returns are relative to player ID. 
    # If bot1 is P0, we want returns[0].
    returns = state.returns()
    if returns[0] > 0: return 1.0   # Bot1 (P0) wins
    if returns[1] > 0: return 0.0   # Bot2 (P1) wins
    return 0.5                      # Draw

def run_experiment(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Parameter Grid ---
    ig_depths = [2, 3, 4]
    mcts_iterations = [5, 10, 50, 100, 250, 500, 1000, 2000]
    
    # Games to test against (from config)
    games_config = config["games"]
    
    # Storage for results
    # Z-axis matrix: [Depth_Index][MCTS_Index]
    win_rates = np.zeros((len(ig_depths), len(mcts_iterations)))

    print(f"Starting Manifold Experiment...")
    print(f"Depths: {ig_depths}")
    print(f"MCTS Iters: {mcts_iterations}")
    print(f"Variants: {len(games_config)}")

    for d_idx, depth in enumerate(ig_depths):
        for m_idx, mcts_k in enumerate(mcts_iterations):
            
            total_wins = 0
            total_games = 0
            
            print(f"\nEvaluating: IG(Depth={depth}) vs MCTS(k={mcts_k})")
            
            # Iterate over all game variants in the yaml
            for g_cfg in games_config:
                game = load_game_safe(g_cfg)
                
                # Instantiate Agents
                # Note: We create fresh agents for each game to ensure state/rules match
                default_weights = {'connect': 1.0, 'block': 1.0, 'center': 1.0}
                # Agent 1: Depth Limited Intuitive Gamer
                ig_bot = DepthLimitedIGPolicy(game, player_id=0, max_depth=depth, k=3, weights=default_weights) 
                
                # Agent 2: MCTS
                mcts_bot = MCTSAgent(game, player_id=1, iterations=mcts_k)
                
                # Run Trials (e.g., 20 games per variant to get an average)
                trials_per_variant = 10 
                for _ in range(trials_per_variant):
                    # We alternate who starts to be fair, but track IG's win rate
                    # Game 1: IG starts (P0)
                    res = play_match(game, ig_bot, mcts_bot)
                    if res != 0.5:
                        total_wins += res
                        total_games += 1
                    
                    # Game 2: MCTS starts (P0), IG is P1
                    # Note: We need to re-init bots with correct player_ids if they rely on it
                    ig_bot_p1 = DepthLimitedIGPolicy(game, player_id=1, max_depth=depth, k=4, weights=default_weights)
                    mcts_bot_p0 = MCTSAgent(game, player_id=0, iterations=mcts_k)
                    
                    res_flipped = play_match(game, mcts_bot_p0, ig_bot_p1)
                    # Result from perspective of P0 (MCTS). 
                    # If P0 wins (val=1), IG lost. If P0 loses (val=0), IG won.
                    if res_flipped != 0.5:
                        # If res_flipped is 1.0 (MCTS wins), we add 0.0 to total_wins
                        # If res_flipped is 0.0 (IG wins), we add 1.0 to total_wins
                        total_wins += (1.0 - res_flipped)
                        total_games += 1
                    
            if total_games > 0:
                avg_win_rate = total_wins / total_games
            else:
                avg_win_rate = 0.0
            win_rates[d_idx, m_idx] = avg_win_rate
            print(f"  -> Win Rate: {avg_win_rate:.2f}")

    return ig_depths, mcts_iterations, win_rates

# --- 3. Visualization ---
def plot_manifold(depths, mcts_iters, win_rates):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid
    # X: Depths (Linear)
    # Y: MCTS Iterations (We use indices 0..N for equidistant tickers, then label them)
    X, Y = np.meshgrid(depths, np.arange(len(mcts_iters)))
    
    # Transpose win_rates to match meshgrid (Y, X) shape requirements
    Z = win_rates.T 

    # Plot Surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='k', alpha=0.9, linewidth=0.5)

    # --- Hyperplane at Z=0.5 ---
    # Create a light plane spanning the full XY range at Z=0.5
    xx, yy = np.meshgrid([min(depths), max(depths)], [0, len(mcts_iters)-1])
    zz = np.full_like(xx, 0.5, dtype=float)
    ax.plot_surface(xx, yy, zz, color='red', alpha=0.2) 
    # Add text label for the plane
    ax.text(min(depths), 0, 0.5, "Equivalence (50%)", color='red')

    # --- Formatting ---
    ax.set_xlabel('Intuitive Gamer Depth')
    ax.set_ylabel('MCTS Iterations')
    ax.set_zlabel('IG Win Rate')
    
    # Set Y-ticks to be equidistant (0, 1, 2...) but labeled with actual MCTS counts
    ax.set_yticks(np.arange(len(mcts_iters)))
    ax.set_yticklabels([str(k) for k in mcts_iters])
    
    # Set X-ticks to integers
    ax.set_xticks(depths)

    ax.set_title('Performance Manifold: Intuitive Gamer vs MCTS')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Win Rate')

    plt.savefig("ig_vs_mcts_manifold.png")
    print("Plot saved to ig_vs_mcts_manifold.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    depths, iters, rates = run_experiment(args.config)
    plot_manifold(depths, iters, rates)