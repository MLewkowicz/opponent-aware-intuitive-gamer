#!/usr/bin/env python3
"""
Simulated Gameplay Framework

This module provides functionality to simulate games between two policies
across multiple game variants and collect statistics on game outcomes.
"""

import argparse
import yaml
import statistics
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import random
import numpy as np
import matplotlib.pyplot as plt

import pyspiel
from utils.utils import load_game, load_policies
import copy

@dataclass
class GameResult:
    """Result of a single game simulation."""
    winner: int  # 0, 1, or -1 for draw (player position)
    policy_winner: int  # 0, 1, or -1 for draw (which policy won)
    num_turns: int
    final_utility: List[float]  # Utility for each player
    policies_swapped: bool  # Whether policies were swapped for this game
    posteriors_over_time: Optional[Dict[str, List[List[float]]]] = None  # policy_name -> [turn_posteriors]


@dataclass
class SimulationStats:
    """Statistics for a set of game simulations."""
    game_name: str
    policy_0_name: str
    policy_1_name: str
    total_games: int
    
    # Policy-based wins (who actually won, regardless of player position)
    policy_0_wins: int  # How many times policy 0 won
    policy_1_wins: int  # How many times policy 1 won
    draws: int
    
    # Position-based performance (how each policy performed as player 1 vs player 2)
    policy_0_as_p1_wins: int  # Policy 0 wins when playing as player 1
    policy_0_as_p1_games: int # Policy 0 games as player 1
    policy_0_as_p2_wins: int  # Policy 0 wins when playing as player 2  
    policy_0_as_p2_games: int # Policy 0 games as player 2
    
    policy_1_as_p1_wins: int  # Policy 1 wins when playing as player 1
    policy_1_as_p1_games: int # Policy 1 games as player 1
    policy_1_as_p2_wins: int  # Policy 1 wins when playing as player 2
    policy_1_as_p2_games: int # Policy 1 games as player 2
    
    avg_turns: float
    std_turns: float
    min_turns: int
    max_turns: int
    
    @property
    def policy_0_win_rate(self) -> float:
        return self.policy_0_wins / self.total_games if self.total_games > 0 else 0.0
    
    @property 
    def policy_1_win_rate(self) -> float:
        return self.policy_1_wins / self.total_games if self.total_games > 0 else 0.0
        
    @property
    def draw_rate(self) -> float:
        return self.draws / self.total_games if self.total_games > 0 else 0.0
        
    @property
    def policy_0_as_p1_win_rate(self) -> float:
        return self.policy_0_as_p1_wins / self.policy_0_as_p1_games if self.policy_0_as_p1_games > 0 else 0.0
        
    @property
    def policy_0_as_p2_win_rate(self) -> float:
        return self.policy_0_as_p2_wins / self.policy_0_as_p2_games if self.policy_0_as_p2_games > 0 else 0.0
        
    @property
    def policy_1_as_p1_win_rate(self) -> float:
        return self.policy_1_as_p1_wins / self.policy_1_as_p1_games if self.policy_1_as_p1_games > 0 else 0.0
        
    @property
    def policy_1_as_p2_win_rate(self) -> float:
        return self.policy_1_as_p2_wins / self.policy_1_as_p2_games if self.policy_1_as_p2_games > 0 else 0.0


class GameSimulator:
    """Simulates gameplay between two policies."""
    
    def __init__(self, policy_0, policy_1, game: pyspiel.Game):
        self.policy_0 = policy_0
        self.policy_1 = policy_1
        self.game = game
        self.policies = [policy_0, policy_1]
    
    def simulate_game(self, alternate_starting_player: bool = False, game_index: int = 0, track_posteriors: bool = False) -> GameResult:
        """Simulate a single game between the two policies.
        
        Args:
            alternate_starting_player: If True, alternate who goes first based on game_index
            game_index: Index of current game (used for alternating starting player)
            track_posteriors: If True, track posterior probabilities over time
            
        Returns:
            GameResult with winner, number of turns, and final utilities
        """
        state = self.game.new_initial_state()
        turn_count = 0
        
        # Initialize posterior tracking
        posteriors_over_time = {} if track_posteriors else None
        
        # Determine starting player assignment
        if alternate_starting_player and game_index % 2 == 1:
            # Swap policies for this game
            current_policies = [self.policies[1], self.policies[0]]
        else:
            current_policies = self.policies[:]
        
        for policy in current_policies:
            policy.assign_playerid(current_policies.index(policy))
            policy.reset()
        
        # Initialize posterior tracking for policies with posteriors
        if track_posteriors and posteriors_over_time is not None:
            for i, policy_wrapper in enumerate(current_policies):
                policy_obj = policy_wrapper['policy'] if hasattr(policy_wrapper, '__getitem__') else policy_wrapper
                if hasattr(policy_obj, 'posteriors'):
                    policy_name = policy_wrapper.get('name', f'policy_{i}') if hasattr(policy_wrapper, '__getitem__') else f'policy_{i}'
                    posteriors_over_time[policy_name] = []


        while not state.is_terminal():
            current_player = state.current_player()
            
            # Get action from current player's policy
            if current_player >= 0 and current_player < len(current_policies):
                policy = current_policies[current_player]
                if hasattr(policy, 'policy'):
                    # Handle wrapped policy objects from load_policies
                    policy_obj = policy['policy']
                else:
                    policy_obj = policy
                
                # Get action probabilities and sample
                action = policy_obj.step(state)                    
                for policy in current_policies:
                    policy.update_action_choices(action, copy.deepcopy(state), current_player)

            else:
                # Fallback for unexpected player indices
                print("Unexpected player index, selecting random action.")
                legal_actions = state.legal_actions()
                action = random.choice(legal_actions) if legal_actions else 0
            
                
            state.apply_action(action)
            turn_count += 1
            
            # Track posteriors after each action
            if track_posteriors and posteriors_over_time is not None:
                for i, policy_wrapper in enumerate(current_policies):
                    policy_obj = policy_wrapper['policy'] if hasattr(policy_wrapper, '__getitem__') else policy_wrapper
                    if hasattr(policy_obj, 'posteriors'):
                        policy_name = policy_wrapper.get('name', f'policy_{i}') if hasattr(policy_wrapper, '__getitem__') else f'policy_{i}'
                        if policy_name in posteriors_over_time:
                            # Get current posteriors and store a copy
                            current_posteriors = getattr(policy_obj, 'posteriors', [])
                            posteriors_over_time[policy_name].append(copy.deepcopy(current_posteriors))
            
            # Safety check for infinite games
            if turn_count > 1000:
                print(f"Warning: Game exceeded 1000 turns, terminating...")
                break
        
        # Determine winner and get utilities
        final_utilities = state.returns()
        policies_swapped = alternate_starting_player and game_index % 2 == 1
        
        if len(final_utilities) >= 2:
            if final_utilities[0] > final_utilities[1]:
                winner = 0  # Player 0 won
            elif final_utilities[1] > final_utilities[0]:
                winner = 1  # Player 1 won
            else:
                winner = -1  # Draw
        else:
            winner = -1  # Draw/unknown
        
        # Determine which policy actually won
        if winner == -1:
            policy_winner = -1  # Draw
        elif policies_swapped:
            # If policies were swapped, policy 0 was playing as player 1, policy 1 as player 0
            policy_winner = 1 if winner == 0 else 0
        else:
            # Normal assignment: policy 0 as player 0, policy 1 as player 1  
            policy_winner = winner
        
        # Keep original utilities (don't swap them back)
        return GameResult(
            winner=winner,
            policy_winner=policy_winner,
            num_turns=turn_count,
            final_utility=final_utilities,
            policies_swapped=policies_swapped,
            posteriors_over_time=posteriors_over_time
        )
    
    def simulate_multiple_games(self, num_trials: int, alternate_starting_player: bool = True, track_posteriors: bool = False) -> List[GameResult]:
        """Simulate multiple games and return results."""
        results = []
        
        for i in range(num_trials):
            try:
                result = self.simulate_game(alternate_starting_player, i, track_posteriors)
                results.append(result)
            except Exception as e:
                print(f"Warning: Game {i+1} failed with error: {e}")
                continue
                
        return results


def plot_posteriors_over_time(results: List[GameResult], policy_names: List[str], save_path: Optional[str] = None) -> None:
    """Plot average posterior probabilities of opponent types over game turns for intuitive gamer."""
    
    # Look for intuitive gamer policies and collect their opponent belief data
    opponent_beliefs = {}  # opponent_type -> [turn_data_across_games]
    
    print(f"Debug: Processing {len(results)} results for posterior plotting")
    
    for result_idx, result in enumerate(results):
        if result.posteriors_over_time is None:
            continue
            
        print(f"Debug: Result {result_idx} has posteriors for policies: {list(result.posteriors_over_time.keys())}")
        
        # Find policies with opponent inference (any policy that has posteriors)
        for policy_name, turn_posteriors in result.posteriors_over_time.items():
            print(f"Debug: Policy {policy_name} has {len(turn_posteriors)} turns of posteriors")
            
            if len(turn_posteriors) > 0:
                print(f"Debug: First few posteriors: {turn_posteriors[:2]}")
                
                # Process this policy's opponent beliefs
                for turn_idx, turn_posterior_list in enumerate(turn_posteriors):
                    if isinstance(turn_posterior_list, list) and len(turn_posterior_list) > 0:
                        # Each turn contains a list of posterior dicts
                        # Take the last posterior from each turn (final belief after all actions in that turn)
                        final_posterior = turn_posterior_list[-1]
                        
                        if isinstance(final_posterior, dict):
                            # final_posterior should be like {'random': 0.7, 'intuitive_gamer': 0.3}
                            for opponent_type, prob_value in final_posterior.items():
                                if opponent_type not in opponent_beliefs:
                                    opponent_beliefs[opponent_type] = []
                                
                                # Ensure we have enough turn slots
                                while len(opponent_beliefs[opponent_type]) <= turn_idx:
                                    opponent_beliefs[opponent_type].append([])
                                
                                try:
                                    prob = float(prob_value)
                                    opponent_beliefs[opponent_type][turn_idx].append(prob)
                                except (TypeError, ValueError):
                                    print(f"Debug: Could not convert {opponent_type} prob {prob_value} to float")
                                    continue
                        else:
                            print(f"Debug: Turn {turn_idx} final posterior is not a dict: {type(final_posterior)}")
                    elif isinstance(turn_posterior_list, dict):
                        # Direct dict case (shouldn't happen based on debug, but just in case)
                        for opponent_type, prob_value in turn_posterior_list.items():
                            if opponent_type not in opponent_beliefs:
                                opponent_beliefs[opponent_type] = []
                            
                            while len(opponent_beliefs[opponent_type]) <= turn_idx:
                                opponent_beliefs[opponent_type].append([])
                            
                            try:
                                prob = float(prob_value)
                                opponent_beliefs[opponent_type][turn_idx].append(prob)
                            except (TypeError, ValueError):
                                continue
                    else:
                        print(f"Debug: Turn {turn_idx} posteriors structure unexpected: {type(turn_posterior_list)}")
            
            # Only process the first policy with posteriors for now
            if opponent_beliefs:
                break
        
        # Only process first few results for debugging
        if result_idx >= 2:
            break
    
    print(f"Debug: Collected opponent beliefs for: {list(opponent_beliefs.keys())}")
    for opponent_type, turn_data in opponent_beliefs.items():
        print(f"Debug: {opponent_type} has data for {len(turn_data)} turns")
    
    if not opponent_beliefs:
        print("No opponent belief data found to plot.")
        print("Make sure you have an intuitive_gamer policy with opponent_inference enabled.")
        return
    
    # Create a single plot showing opponent beliefs over time
    plt.figure(figsize=(12, 8))
    
    # Plot each opponent type
    for opponent_type, turn_data in opponent_beliefs.items():
        # Calculate mean and std for each turn
        turns = []
        means = []
        stds = []
        
        for turn_idx, probs in enumerate(turn_data):
            if len(probs) > 0:
                turns.append(turn_idx)
                means.append(np.mean(probs))
                stds.append(np.std(probs))
        
        if len(turns) > 0:
            turns = np.array(turns)
            means = np.array(means)
            stds = np.array(stds)
            
            # Plot mean line
            plt.plot(turns, means, label=f'Belief opponent is {opponent_type}', linewidth=3, marker='o', markersize=4)
            
            # Add std shading
            plt.fill_between(turns, means - stds, means + stds, alpha=0.2)
            
            # Print some debug info
            print(f"Opponent type '{opponent_type}':")
            print(f"  Turn range: 0 to {len(turns)-1}")
            print(f"  Belief range: {np.min(means):.3f} to {np.max(means):.3f}")
            print(f"  Average std: {np.mean(stds):.3f}")
            print(f"  Games per turn: {[len(probs) for probs in turn_data[:5]]} (first 5 turns)")
    
    plt.title('Intuitive Gamer: Belief About Opponent Type Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Turn Number', fontsize=12)
    plt.ylabel('Posterior Probability', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add a horizontal line at 0.5 for reference
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal belief (0.5)')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Posterior plot saved to {save_path}")
    
    plt.show()


def calculate_stats(results: List[GameResult], game_name: str, policy_0_name: str, policy_1_name: str) -> SimulationStats:
    """Calculate statistics from simulation results."""
    if not results:
        return SimulationStats(
            game_name=game_name,
            policy_0_name=policy_0_name,
            policy_1_name=policy_1_name,
            total_games=0,
            policy_0_wins=0,
            policy_1_wins=0,
            draws=0,
            policy_0_as_p1_wins=0,
            policy_0_as_p1_games=0,
            policy_0_as_p2_wins=0,
            policy_0_as_p2_games=0,
            policy_1_as_p1_wins=0,
            policy_1_as_p1_games=0,
            policy_1_as_p2_wins=0,
            policy_1_as_p2_games=0,
            avg_turns=0.0,
            std_turns=0.0,
            min_turns=0,
            max_turns=0
        )
    
    total_games = len(results)
    policy_0_wins = sum(1 for r in results if r.policy_winner == 0)
    policy_1_wins = sum(1 for r in results if r.policy_winner == 1)
    draws = sum(1 for r in results if r.policy_winner == -1)
    
    # Calculate position-based performance
    policy_0_as_p1_games = sum(1 for r in results if r.policies_swapped)  # When swapped, policy 0 plays as player 1
    policy_0_as_p2_games = sum(1 for r in results if not r.policies_swapped)  # When not swapped, policy 0 plays as player 0 (which is player 2 in 0-indexed)
    
    policy_1_as_p1_games = sum(1 for r in results if not r.policies_swapped)  # When not swapped, policy 1 plays as player 1
    policy_1_as_p2_games = sum(1 for r in results if r.policies_swapped)  # When swapped, policy 1 plays as player 0 (which is player 2 in 0-indexed)
    
    policy_0_as_p1_wins = sum(1 for r in results if r.policies_swapped and r.policy_winner == 0)
    policy_0_as_p2_wins = sum(1 for r in results if not r.policies_swapped and r.policy_winner == 0)
    
    policy_1_as_p1_wins = sum(1 for r in results if not r.policies_swapped and r.policy_winner == 1)  
    policy_1_as_p2_wins = sum(1 for r in results if r.policies_swapped and r.policy_winner == 1)
    
    turn_counts = [r.num_turns for r in results]
    avg_turns = statistics.mean(turn_counts)
    std_turns = statistics.stdev(turn_counts) if len(turn_counts) > 1 else 0.0
    min_turns = min(turn_counts)
    max_turns = max(turn_counts)
    
    return SimulationStats(
        game_name=game_name,
        policy_0_name=policy_0_name,
        policy_1_name=policy_1_name,
        total_games=total_games,
        policy_0_wins=policy_0_wins,
        policy_1_wins=policy_1_wins,
        draws=draws,
        policy_0_as_p1_wins=policy_0_as_p1_wins,
        policy_0_as_p1_games=policy_0_as_p1_games,
        policy_0_as_p2_wins=policy_0_as_p2_wins,
        policy_0_as_p2_games=policy_0_as_p2_games,
        policy_1_as_p1_wins=policy_1_as_p1_wins,
        policy_1_as_p1_games=policy_1_as_p1_games,
        policy_1_as_p2_wins=policy_1_as_p2_wins,
        policy_1_as_p2_games=policy_1_as_p2_games,
        avg_turns=avg_turns,
        std_turns=std_turns,
        min_turns=min_turns,
        max_turns=max_turns
    )


def run_simulation_experiment(config: Dict[str, Any]) -> None:
    """Run the complete simulation experiment from config."""
    print("="*60)
    print("SIMULATED GAMEPLAY EXPERIMENT")
    print("="*60)
    
    # Extract configuration
    policies_config = config["policies"]
    games_config = config["games"]
    num_trials = config.get("num_trials", 100)
    alternate_starting = config.get("alternate_starting_player", True)
    
    if len(policies_config) != 2:
        raise ValueError(f"Expected exactly 2 policies, got {len(policies_config)}")
    
    print(f"Number of trials per game: {num_trials}")
    print(f"Alternate starting player: {alternate_starting}")
    print(f"Policy 1: {policies_config[0]['name']}")
    print(f"Policy 2: {policies_config[1]['name']}")
    print(f"Games to simulate: {len(games_config)}")
    
    all_stats = []
    
    # Run simulations for each game
    for game_config in games_config:
        print(f"\n{'-'*40}")
        print(f"Simulating game: {game_config['name']}")
        
        try:
            # Load game
            game = load_game(game_config)
            
            # Generate game name for display
            params = game_config.get("parameters", {})
            if game_config["name"] == "mnk_game":
                m, n, k = params.get('m'), params.get('n'), params.get('k')
                rules = params.get('rules', {})
                variant_tag = "standard"
                if rules.get("allowed_directions"): variant_tag = "restricted"
                if rules.get("p0_extra_k"): variant_tag = "asymmetric_k"
                if rules.get("opening_moves"): variant_tag = "opening_moves"
                game_display_name = f"mnk_{m}x{n}_k{k}_{variant_tag}"
            else:
                game_display_name = game_config["name"]
            
            # Load policies
            policies = load_policies(policies_config, game, include_metadata=False)
            policy_list = list(policies.values())
            policy_names = [policies_config[0]['name'], policies_config[1]['name']]
            
            if len(policy_list) < 2:
                print(f"Error: Could not load 2 policies for {game_display_name}")
                continue
            
            # Create simulator
            simulator = GameSimulator(policy_list[0], policy_list[1], game)
            
            # Check if we should track posteriors (only for single game experiments)
            track_posteriors = len(games_config) == 1 and config.get("track_posteriors", False)
            if track_posteriors:
                print("Posterior tracking enabled (single game experiment)")
            
            # Run simulations
            print(f"Running {num_trials} games...")
            results = simulator.simulate_multiple_games(num_trials, alternate_starting, track_posteriors)
            
            # Calculate statistics
            stats = calculate_stats(results, game_display_name, policy_names[0], policy_names[1])
            all_stats.append(stats)
            
            # Print results for this game
            print(f"✓ Completed {stats.total_games}/{num_trials} games")
            print(f"  {stats.policy_0_name} wins: {stats.policy_0_wins} ({stats.policy_0_win_rate:.1%})")
            print(f"  {stats.policy_1_name} wins: {stats.policy_1_wins} ({stats.policy_1_win_rate:.1%})")
            print(f"  Draws: {stats.draws} ({stats.draw_rate:.1%})")
            print(f"  {stats.policy_0_name} as P1: {stats.policy_0_as_p1_win_rate:.1%} ({stats.policy_0_as_p1_wins}/{stats.policy_0_as_p1_games})")
            print(f"  {stats.policy_0_name} as P2: {stats.policy_0_as_p2_win_rate:.1%} ({stats.policy_0_as_p2_wins}/{stats.policy_0_as_p2_games})")
            print(f"  {stats.policy_1_name} as P1: {stats.policy_1_as_p1_win_rate:.1%} ({stats.policy_1_as_p1_wins}/{stats.policy_1_as_p1_games})")
            print(f"  {stats.policy_1_name} as P2: {stats.policy_1_as_p2_win_rate:.1%} ({stats.policy_1_as_p2_wins}/{stats.policy_1_as_p2_games})")
            print(f"  Avg turns: {stats.avg_turns:.2f} ± {stats.std_turns:.2f}")
            print(f"  Turn range: {stats.min_turns} - {stats.max_turns}")
            
            # Plot posteriors if tracked
            if track_posteriors and any(r.posteriors_over_time for r in results):
                print("\nGenerating posterior probability plots...")
                plot_posteriors_over_time(results, [stats.policy_0_name, stats.policy_1_name])
            
        except Exception as e:
            print(f"✗ Failed to simulate {game_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print('='*60)
    
    if all_stats:
        # Overall statistics across all games
        total_games = sum(s.total_games for s in all_stats)
        total_p0_wins = sum(s.policy_0_wins for s in all_stats)
        total_p1_wins = sum(s.policy_1_wins for s in all_stats)
        total_draws = sum(s.draws for s in all_stats)
        
        # Weighted average of turns
        all_turns = []
        for stats in all_stats:
            # Approximate turn counts based on avg ± std
            game_turns = [stats.avg_turns] * stats.total_games
            all_turns.extend(game_turns)
        
        if all_turns:
            overall_avg_turns = statistics.mean(all_turns)
            overall_std_turns = statistics.stdev(all_turns) if len(all_turns) > 1 else 0.0
        else:
            overall_avg_turns = 0.0
            overall_std_turns = 0.0
            
        # Get policy names from first stats entry
        policy_0_name = all_stats[0].policy_0_name if all_stats else "Policy 0"
        policy_1_name = all_stats[0].policy_1_name if all_stats else "Policy 1"
        
        print(f"Total games played: {total_games}")
        print(f"Overall {policy_0_name} win rate: {total_p0_wins/total_games:.1%} ({total_p0_wins} wins)")
        print(f"Overall {policy_1_name} win rate: {total_p1_wins/total_games:.1%} ({total_p1_wins} wins)")
        print(f"Overall draw rate: {total_draws/total_games:.1%} ({total_draws} draws)")
        print(f"Overall average turns: {overall_avg_turns:.2f} ± {overall_std_turns:.2f}")
        
        print(f"\nPer-game breakdown:")
        print(f"{'Game':<25} {'Games':<6} {f'{policy_0_name} Win%':<12} {f'{policy_1_name} Win%':<12} {'Draw%':<6} {'Avg Turns':<10}")
        print("-" * 80)
        for stats in all_stats:
            print(f"{stats.game_name:<25} {stats.total_games:<6} "
                  f"{stats.policy_0_win_rate:<12.1%} {stats.policy_1_win_rate:<12.1%} {stats.draw_rate:<6.1%} "
                  f"{stats.avg_turns:.1f}±{stats.std_turns:.1f}")
    else:
        print("No successful simulations completed.")


def main():
    """Main entry point for simulated gameplay experiments."""
    parser = argparse.ArgumentParser(description="Run simulated gameplay experiments")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to simulation configuration file")
    
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        run_simulation_experiment(config)
        
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found")
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration file: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
