import argparse
import yaml
import numpy as np
import pyspiel
import pandas as pd
import itertools
from typing import List, Dict, Any

from policies.policy_registry import POLICY_REGISTRY, instantiate_policy
from policies.intuitivegamer.policy import IntuitiveGamerPolicy

def load_game(game_config: Dict[str, Any]) -> pyspiel.Game:
    """Load and initialize a game from configuration."""
    game_name = game_config["name"]
    game_params = game_config.get("parameters", {})
    
    try:
        if game_params:
            game = pyspiel.load_game(game_name, game_params)
        else:
            game = pyspiel.load_game(game_name)
        print(f"✓ Loaded game: {game_name}")
        return game
    except Exception as e:
        raise RuntimeError(f"Failed to load game '{game_name}': {e}")

def load_policies(policies_config: List[Dict[str, Any]], game: pyspiel.Game) -> Dict[str, Any]:
    """Load and initialize policies from configuration."""
    policies = {}
    print(f"Loading opponent policies from config:")
    
    for i, policy_config in enumerate(policies_config):
        try:
            # Add the game parameter to policy config
            policy_config_with_game = policy_config.copy()
            if "parameters" not in policy_config_with_game:
                policy_config_with_game["parameters"] = {}
            policy_config_with_game["parameters"]["game"] = game
            
            policy = instantiate_policy(policy_config_with_game)
            policy_name = policy_config["name"]
            
            # Handle duplicate names in config (e.g. two mcts agents)
            policy_id = policy_name
            count = 1
            while policy_id in policies:
                policy_id = f"{policy_name}_{count}"
                count += 1
        
            policies[policy_id] = policy
            print(f"  ✓ Loaded Opponent: {policy_id}")
            
        except Exception as e:
            print(f"  ✗ Failed to load policy '{policy_config.get('name')}': {e}")
            continue
    
    if not policies:
        raise RuntimeError("No opponent policies were successfully loaded.")
    
    return policies

def get_random_start_states(game, num_states=50, max_random_moves=6):
    """Generates a list of valid, non-terminal starting states."""
    states = []
    # Use a fixed seed for board generation reproducibility
    rng = np.random.RandomState(42) 
    
    for _ in range(num_states):
        state = game.new_initial_state()
        n_moves = rng.randint(0, max_random_moves + 1)
        
        valid_state = True
        for _ in range(n_moves):
            if state.is_terminal():
                valid_state = False
                break
            actions = state.legal_actions()
            a = rng.choice(actions)
            state.apply_action(a)
            
        if valid_state and not state.is_terminal():
            states.append(state.clone())
        else:
            states.append(game.new_initial_state())
    return states

def play_match(game, state, bot1, bot2):
    """
    Plays a game starting from 'state' between bot1 (current player) and bot2 (next).
    Returns return for bot1.
    """
    curr_state = state.clone()
    p1_id = curr_state.current_player()
    p2_id = 1 - p1_id
    
    bots = {p1_id: bot1, p2_id: bot2}
    
    while not curr_state.is_terminal():
        current_player = curr_state.current_player()
        bot = bots[current_player]
        
        # Determine if bot has a 'step' method or needs a wrapper
        if hasattr(bot, 'step'):
            action = bot.step(curr_state)
        else:
            # Fallback if policy doesn't have step (shouldn't happen with new base)
            raise NotImplementedError(f"Policy {type(bot)} must implement .step(state)")

        curr_state.apply_action(action)
        
    return curr_state.returns()[p1_id]

def run_parameter_search(config):
    # 1. Load Game
    game = load_game(config["game"])
    
    # 2. Load Opponents from Config
    # These are the static opponents we evaluate against (e.g. MCTS, Random, BaseIG)
    opponents = load_policies(config["policies"], game)
    
    # 3. Generate Evaluation Set
    print("\nGenerating evaluation states...")
    eval_states = get_random_start_states(game, num_states=50)
    
    # 4. Define Weight Grid
    print("Initializing parameter grid...")
    # 0.0 to 2.0 in 0.2 increments
    weight_range = [round(x, 1) for x in np.arange(0.0, 2.1, 0.2)]
    combinations = list(itertools.product(weight_range, repeat=3))
    
    print(f"Total combinations to test: {len(combinations)}")
    print(f"Opponents: {list(opponents.keys())}")
    
    results = []
    
    # 5. Run Grid Search
    for i, (wc, wb, wa) in enumerate(combinations):
        if i % 100 == 0:
            print(f"Processing combination {i}/{len(combinations)}...")
            
        current_weights = {'connect': wc, 'block': wb, 'center': wa}
        
        # Instantiate Challenger (The Intuitive Gamer we are optimizing)
        challenger_bot = IntuitiveGamerPolicy(game, weights=current_weights)
        
        row = {
            "w_connect": wc, 
            "w_block": wb, 
            "w_center": wa
        }
        
        # Play against each opponent loaded from config
        for opp_name, opp_bot in opponents.items():
            wins = 0
            losses = 0
            draws = 0
            
            for state in eval_states:
                outcome = play_match(game, state, challenger_bot, opp_bot)
                
                if outcome > 0: wins += 1
                elif outcome < 0: losses += 1
                else: draws += 1
            
            total = wins + losses + draws
            win_rate = wins / total if total > 0 else 0.0
            
            # Store stats for this opponent
            row[f"win_rate_vs_{opp_name}"] = win_rate
            row[f"wins_vs_{opp_name}"] = wins
            row[f"losses_vs_{opp_name}"] = losses
            row[f"draws_vs_{opp_name}"] = draws
            
        results.append(row)

    # 6. Format Data
    df = pd.DataFrame(results)
    df.set_index(['w_connect', 'w_block', 'w_center'], inplace=True)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Run Intuitive Gamer parameter grid search")
    parser.add_argument("--config", "-c", type=str, required=True, 
                        help="Path to experiment configuration file (yaml)")
    parser.add_argument("--output", "-o", type=str, default="intuitive_gamer_grid_search.csv",
                        help="Path to save results CSV")

    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        results_df = run_parameter_search(config)
        
        print("\nSearch Complete.")
        
        # Dynamic print of top results based on first opponent found (usually random or mcts)
        first_opp = list(results_df.columns)[0].split('_vs_')[-1] # extract name
        col_name = f"win_rate_vs_{first_opp}"
        
        if col_name in results_df.columns:
            print(f"\nTop 5 configurations vs {first_opp}:")
            print(results_df.sort_values(col_name, ascending=False).head(5))

        results_df.to_csv(args.output)
        print(f"\nResults saved to '{args.output}'")
        
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()