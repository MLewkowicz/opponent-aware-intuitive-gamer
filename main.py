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
    
    print(f"Initial dataset size: {len(view)}")
        
    for p in scfg.get("predicates",[]):
        fn = p if callable(p) else eval(p)
        view = view.where(fn)
        print(f"After predicate: {len(view)} items")

    # read sampling parameters
    sample_cfg = scfg.get("sample", {})
    k = sample_cfg.get("k", 1)
    replace = sample_cfg.get("replace", True)
    
    print(f"Final dataset size: {len(view)}, sampling k={k}, replace={replace}")
    
    # Check if we have enough items for sampling without replacement
    if not replace and k > len(view):
        print(f"Warning: Cannot sample {k} items without replacement from {len(view)} items. Using replacement instead.")
        replace = True
    
    # Test sampling immediately to catch errors early
    print("Testing sample generation...")
    try:
        test_samples = view.sample(k=min(3, len(view)), replace=replace)
        print(f"Test sampling successful: got {len(test_samples)} samples")
        if test_samples:
            print(f"Sample structure: {list(test_samples[0].keys()) if test_samples[0] else 'Empty sample'}")
    except Exception as e:
        print(f"Error during test sampling: {e}")
        return None
    
    # attach sampling function to view    
    view.samples = lambda: view.sample(k=k, replace=replace)

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

    print("\nBuilding sampler...")
    sampler = build_sampler(config, game)
    
    if sampler is None:
        print("Error: Failed to build sampler")
        return
    
    print("Importing analysis modules...")
    from analysis import Experiment, visualize_agreement_heatmap
    
    print("Creating experiment...")
    experiment = Experiment(policies, sampler)

    print("Running pairwise comparison...")
    try:
        experiment = Experiment(policies, sampler)
        exp_one = experiment.run_max_action_disagreement()
        experiment = Experiment(policies, sampler)
        exp_two = experiment.run_pairwise_comparison()
        if exp_one:
            print("Visualizing results...")
            visualize_agreement_heatmap([exp_one, exp_two], titles=["Max Action Agreement", "Pairwise Action Agreement"])
        else:
            print("No results to visualize")
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
    

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
