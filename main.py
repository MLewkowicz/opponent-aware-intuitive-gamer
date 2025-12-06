import yaml
import argparse
import pyspiel
from typing import List, Dict, Any
from policies.policy_registry import instantiate_policy, POLICY_REGISTRY
import os
from state_dataset.dataset import GameStateView, GameStateDataset


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
    
    print(f"\nAvailable policies: {list(POLICY_REGISTRY.keys())}")
    print("Loading policies:")
    
    for i, policy_config in enumerate(policies_config):
        try:
            # Add the game parameter to policy config
            policy_config_with_game = policy_config.copy()
            if "parameters" not in policy_config_with_game:
                policy_config_with_game["parameters"] = {}
            policy_config_with_game["parameters"]["game"] = game
            
            policy = instantiate_policy(policy_config_with_game)
            policy_name = policy_config["name"]
            policy_id = f"{policy_name}_{i}" if policy_name in policies else policy_name
            
            policies[policy_id] = {
                "policy": policy,
                "config": policy_config
            }
            print(f"  ✓ {policy_id}: {policy.__class__.__name__}")
            
        except Exception as e:
            policy_name = policy_config.get("name", "unknown")
            print(f"  ✗ Failed to load policy '{policy_name}': {e}")
            continue
    
    if not policies:
        raise RuntimeError("No policies were successfully loaded")
    
    return policies

def build_sampler(cfg, game):
    ds = GameStateDataset(game)
    scfg = cfg.get("sampler",{})
    view = GameStateView(ds.all())
    if filters := scfg.get("filters"):
        view = view.filter(**filters)
    for p in scfg.get("predicates",[]):
        fn = p if callable(p) else eval(p)
        view = view.where(fn)
    return view

def run_experiment(config: Dict[str, Any]) -> None:
    """Run the full experiment based on configuration."""
    print(f"Running experiment: {config['experiment_name']}")
    print(f"Description: {config['description']}")
    
    # Load game
    print(f"\n{'='*50}")
    print("GAME SETUP")
    print('='*50)
    game = load_game(config["game"])
    
    # Load policies  
    print(f"\n{'='*50}")
    print("POLICY SETUP")
    print('='*50)
    policies = load_policies(config["policies"], game)

    state_samples = build_sampler(config, game)


    for policy_id, policy_info in policies.items():
        policy = policy_info["policy"]
        print(f"\nEvaluating policy: {policy_id}")

        policy.action_likelihoods(game.new_initial_state())
    


def main():
    parser = argparse.ArgumentParser(description="Run game policy comparison experiments")
    parser.add_argument("--config", "-c", type=str, required=True, 
                       help="Path to experiment configuration file")

    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        run_experiment(config)
        
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found")
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration file: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
