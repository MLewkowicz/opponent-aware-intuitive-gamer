import yaml
import argparse
import pyspiel
import os
from typing import List, Dict, Any

# Imports for custom components
from policies.policy_registry import instantiate_policy, POLICY_REGISTRY
from state_dataset.dataset import GameStateView, GameStateDataset
from utils.utils import load_game, load_policies, generate_cache_filename



def build_sampler(cfg, game, cache_file=None):
    # Pass the game instance and cache file to the dataset generator
    ds = GameStateDataset(game, cache_file=cache_file)
    scfg = cfg.get("sampler",{})
    
    view = GameStateView(ds.all())
    
    print(f"Initial dataset size: {len(view)}")
        
    for p in scfg.get("predicates",[]):
        try:
            fn = p if callable(p) else eval(p)
            view = view.where(fn)
        except Exception as e:
            print(f"Warning: Failed to apply predicate '{p}': {e}")

    print(f"After predicates: {len(view)} items")

    # read sampling parameters
    sample_cfg = scfg.get("sample", {})
    k = sample_cfg.get("k", 1)
    replace = sample_cfg.get("replace", True)
    
    print(f"Sampling k={k}, replace={replace}")
    
    # Safety check for sampling
    if not replace and k > len(view):
        print(f"Warning: Cannot sample {k} items without replacement from {len(view)} items. Using replacement instead.")
        replace = True
    
    # Attach sampling function
    view.samples = lambda: view.sample(k=k, replace=replace)

    return view

def run_experiment(config: Dict[str, Any]) -> None:
    """Run the full experiment based on configuration."""
    print(f"Running experiment: {config['experiment_name']}")
    print(f"Description: {config['description']}")
    
    # Check if we have multiple games or single game config
    if "games" in config:
        run_multi_game_experiment(config)
    else:
        run_single_game_experiment(config)

def run_single_game_experiment(config: Dict[str, Any]) -> None:
    """Run experiment with a single game configuration."""
    # 1. Load Game
    print(f"\n{'='*50}")
    print("GAME SETUP")
    print('='*50)
    game = load_game(config["game"])
    
    # Generate cache filename
    cache_file = generate_cache_filename(config["game"])
    print(f"Cache file: {cache_file}")
    
    # 2. Load Policies
    print(f"\n{'='*50}")
    print("POLICY SETUP")
    print('='*50)
    policies = load_policies(config["policies"], game)

    # 3. Build Sampler
    print("\nBuilding sampler...")
    sampler = build_sampler(config, game, cache_file)
    
    if sampler is None or len(sampler) == 0:
        print("Error: Sampler yielded 0 states. Check predicates or game generation.")
        return
    
    print("Importing analysis modules...")
    from analysis import Experiment, visualize_agreement_heatmap
    
    print("Creating experiment...")
    
    try:
        # FIX: Pass 'policies' directly (the dict of dicts)
        # analysis.py expects to access policy['policy'], so we must not strip the wrapper.
        exp = Experiment(policies, sampler)
        exp_one = exp.run_max_action_disagreement()
        
        # Run Pairwise Comparison
        exp = Experiment(policies, sampler)
        exp_two = exp.run_pairwise_comparison()
        
        if exp_one:
            print("Visualizing results...")
            visualize_agreement_heatmap([exp_one, exp_two], titles=["Max Action Agreement", "Pairwise Action Agreement"])
        else:
            print("No results to visualize")
            
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()

def run_multi_game_experiment(config: Dict[str, Any]) -> None:
    """Run experiment across multiple games and combine results."""
    print(f"\n{'='*50}")
    print("MULTI-GAME EXPERIMENT")
    print('='*50)
    
    all_results = []
    game_names = []
    
    for i, game_config in enumerate(config["games"]):
        print(f"\n--- Processing Game {i+1}/{len(config['games'])} ---")
        
        # 1. Load Game
        game = load_game(game_config)
        cache_file = generate_cache_filename(game_config)
        print(f"Cache file: {cache_file}")
        
        # Generate descriptive name for this game variant
        params = game_config.get("parameters", {})
        if game_config["name"] == "mnk_game":
            m, n, k = params.get('m'), params.get('n'), params.get('k')
            rules = params.get('rules', {})
            variant_tag = "standard"
            if rules.get("allowed_directions"): variant_tag = "restricted"
            if rules.get("p0_extra_k"): variant_tag = "asymmetric_k"
            if rules.get("opening_moves"): variant_tag = "opening_moves"
            game_name = f"mnk_{m}x{n}_k{k}_{variant_tag}"
        else:
            game_name = game_config["name"]
        
        game_names.append(game_name)
        
        # 2. Load Policies (same for all games)
        policies = load_policies(config["policies"], game)
        
        # 3. Build Sampler
        sampler = build_sampler(config, game, cache_file)
        
        if sampler is None or len(sampler) == 0:
            print(f"Warning: Sampler for game {game_name} yielded 0 states. Skipping.")
            continue
        
        # 4. Run Analysis
        print("Running analysis...")
        from analysis import Experiment, visualize_agreement_heatmap
        
        try:
            exp = Experiment(policies, sampler)
            max_disagreement_result = exp.run_max_action_disagreement()
            exp = Experiment(policies, sampler)
            pairwise_agreement_result = exp.run_pairwise_comparison()
            
            all_results.append({
                "max_action_disagreement": max_disagreement_result,
                "pairwise_agreement": pairwise_agreement_result
            })
            print(f"✓ Completed analysis for {game_name}")
            
        except Exception as e:
            print(f"✗ Failed to analyze {game_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 5. Combine and visualize results
    if all_results:
        print(f"\n{'='*50}")
        print("COMBINED RESULTS")
        print('='*50)
        
        # Separate results by analysis type
        max_disagreement_results = [r["max_action_disagreement"] for r in all_results]
        pairwise_agreement_results = [r["pairwise_agreement"] for r in all_results]
        
        # Combine results based on strategy
        combination_strategy = config.get("combination_strategy", {"method": "concatenate"})
        combined_max_disagreement = combine_game_results(max_disagreement_results, combination_strategy)
        combined_pairwise_agreement = combine_game_results(pairwise_agreement_results, combination_strategy)
        
        print("Visualizing combined agreement heatmaps...")
        from analysis import visualize_agreement_heatmap
        
        # Show two combined plots
        visualize_agreement_heatmap(
            [combined_max_disagreement, combined_pairwise_agreement], 
            titles=["Max Action Agreement", "Pairwise Agreement"]
        )
        
        print("Experiment completed successfully!")
    else:
        print("No valid results obtained from any game variant.")

def combine_game_results(results_list: List[Dict], strategy: Dict[str, Any]) -> Dict:
    """Combine results from multiple games based on the specified strategy."""
    method = strategy.get("method", "simple_average")
    
    if method == "simple_average":
        # Average agreement scores across games
        combined = {}
        for results in results_list:
            for pair_key, stats in results.items():
                if pair_key not in combined:
                    combined[pair_key] = {"agreements": 0, "totals": 0, "count": 0}
                
                if stats["totals"] > 0:
                    score = stats["agreements"] / stats["totals"]
                    combined[pair_key]["agreements"] += score
                    combined[pair_key]["count"] += 1
        
        # Normalize by count
        for pair_key in combined:
            if combined[pair_key]["count"] > 0:
                combined[pair_key]["agreements"] /= combined[pair_key]["count"]
                combined[pair_key]["totals"] = 1  # Normalized
        
        return combined
    
    elif method == "concatenate":
        # Simply concatenate all results
        combined = {}
        for results in results_list:
            for pair_key, stats in results.items():
                if pair_key not in combined:
                    combined[pair_key] = {"agreements": 0, "totals": 0}
                combined[pair_key]["agreements"] += stats["agreements"]
                combined[pair_key]["totals"] += stats["totals"]
        
        return combined
    
    else:
        # Default to first result
        return results_list[0] if results_list else {}
    

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
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()