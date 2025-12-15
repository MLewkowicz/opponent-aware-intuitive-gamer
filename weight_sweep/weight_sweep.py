import argparse
import yaml
import numpy as np
import pandas as pd
import itertools
import copy

from policies.intuitivegamer.policy import IntuitiveGamerPolicy
from utils.utils import load_game, load_policies, build_sampler



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
            
    # Note: Returning states generated above. 
    # If you prefer empty states for fair comparison, uncomment the line below.
    # return states 
    return [game.new_initial_state()] * 50

def play_match(game, state, bot1, bot2):
    """
    Plays a game starting from 'state' between bot1 (current player) and bot2 (next).
    Returns (return for bot1, total_moves_played).
    """
    curr_state = state.clone()
    p1_id = curr_state.current_player()
    p2_id = 1 - p1_id
    
    bots = {p1_id: bot1, p2_id: bot2}
    bot1.assign_playerid(p1_id)
    bot2.assign_playerid(p2_id)
    
    while not curr_state.is_terminal():
        current_player = curr_state.current_player()
        bot = bots[current_player]
        
        # Determine if bot has a 'step' method
        if hasattr(bot, 'step'):
            action = bot.step(curr_state)
        else:
            raise NotImplementedError(f"Policy {type(bot)} must implement .step(state)")

        for bot in bots.values():
            bot.update_action_choices(action, copy.deepcopy(curr_state), current_player)

        curr_state.apply_action(action)
    
    # Return result AND the total move count (game length)
    return curr_state.returns()[p1_id], curr_state.move_number()

def run_parameter_search(config):
    # 1. Load Game
    game = load_game(config["game"])
    
    # 2. Load Opponents from Config
    opponents = load_policies(config["policies"], game)
    
    # 3. Generate Evaluation Set
    print("\nGenerating evaluation states...")
    eval_states = get_random_start_states(game, num_states=50)
    
    # 4. Define Weight Grid
    print("Initializing parameter grid...")
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
            win_turns = []  # List to store turn counts for wins
            loss_turns = [] # List to store turn counts for losses
            
            for state in eval_states:
                outcome, turns = play_match(game, state, challenger_bot, opp_bot)
                
                if outcome > 0: 
                    wins += 1
                    win_turns.append(turns)
                elif outcome < 0: 
                    losses += 1
                    loss_turns.append(turns)
                else: 
                    draws += 1
            
            total = wins + losses + draws
            win_rate = wins / total if total > 0 else 0.0
            
            # Calculate Averages
            avg_win_turns = np.mean(win_turns) if win_turns else 0.0
            avg_loss_turns = np.mean(loss_turns) if loss_turns else 0.0
            
            # Store stats for this opponent
            row[f"win_rate_vs_{opp_name}"] = win_rate
            row[f"wins_vs_{opp_name}"] = wins
            row[f"losses_vs_{opp_name}"] = losses
            row[f"draws_vs_{opp_name}"] = draws
            row[f"avg_turns_win_vs_{opp_name}"] = round(avg_win_turns, 2)
            row[f"avg_turns_loss_vs_{opp_name}"] = round(avg_loss_turns, 2)
            
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
        
        # Dynamic print of top results based on first opponent found
        if not results_df.empty:
            first_opp_col = [c for c in results_df.columns if "win_rate_vs_" in c][0]
            first_opp = first_opp_col.split('_vs_')[-1]
            
            print(f"\nTop 5 configurations vs {first_opp}:")
            # Show win rate and avg turns to win
            cols_to_show = [first_opp_col, f"avg_turns_win_vs_{first_opp}"]
            print(results_df.sort_values(first_opp_col, ascending=False).head(5)[cols_to_show])

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