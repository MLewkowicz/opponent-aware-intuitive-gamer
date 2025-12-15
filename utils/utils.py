from policies.policy_registry import instantiate_policy, POLICY_REGISTRY
from typing import List, Dict, Any
import pyspiel
from state_dataset.dataset import GameStateView, GameStateDataset
from games.mnk_game import MNKGame




# def load_game(game_config: Dict[str, Any]) -> Any:
#     """Load and initialize a game from configuration."""
#     game_name = game_config["name"]
    
#     try:
#         # Use custom game registry for custom games like mnk_game
#         from games.registry import load_custom_game
#         game = load_custom_game(game_config)
#         print(f"✓ Loaded game: {game_name}")
#         return game
#     except Exception as e:
#         raise RuntimeError(f"Failed to load game '{game_name}': {e}")

def load_game(game_config: Dict[str, Any]) -> pyspiel.Game:
    """Load and initialize a game from configuration."""
    game_name = game_config["name"]
    game_params = game_config.get("parameters", {})
    
    try:
        # --- FIX: Check for custom python games first ---
        if game_name == "mnk_game":
            print(f"✓ Loading Custom Python Game: {game_name}")
            # Unpack parameters (m, n, k, rules) into the constructor
            return MNKGame(**game_params)
        # ------------------------------------------------
        
        # Fallback to standard OpenSpiel C++ games
        if game_params:
            game = pyspiel.load_game(game_name, game_params)
        else:
            game = pyspiel.load_game(game_name)
            
        print(f"✓ Loaded OpenSpiel game: {game_name}")
        return game
        
    except Exception as e:
        raise RuntimeError(f"Failed to load game '{game_name}': {e}")



def load_policies(policies_config: List[Dict[str, Any]], game: Any, include_metadata=True) -> Dict[str, Any]:
    """Load and initialize policies from configuration with two-stage opponent inference setup."""
    policies = {}
    pending_inference_configs = {}
    
    print(f"\nAvailable policies: {list(POLICY_REGISTRY.keys())}")
    print("Loading policies (Stage 1 - without opponent inference):")
    
    # Stage 1: Create all policies WITHOUT opponent inference
    for i, policy_config in enumerate(policies_config):
        try:
            # Add the game parameter to policy config
            policy_config_with_game = policy_config.copy()
            if "parameters" not in policy_config_with_game:
                policy_config_with_game["parameters"] = {}
            policy_config_with_game["parameters"]["game"] = game
            
            # Remove opponent_inference from parameters temporarily
            policy_params = policy_config_with_game["parameters"].copy()
            opponent_inference_config = policy_params.pop('opponent_inference', None)
            policy_config_with_game["parameters"] = policy_params
            
            policy = instantiate_policy(policy_config_with_game)
            policy_name = policy_config["name"]
            policy_id = f"{policy_name}_{i}" if policy_name in policies else policy_name
        
            if include_metadata:
                policies[policy_id] = {
                    "policy": policy,
                    "config": policy_config
                }
            else:
                policies[policy_id] = policy
            
            # Store opponent inference config for later
            if opponent_inference_config:
                pending_inference_configs[policy_id] = opponent_inference_config
                
            print(f"  ✓ {policy_id}: {policy.__class__.__name__}")
            
        except Exception as e:
            policy_name = policy_config.get("name", "unknown")
            print(f"  ✗ Failed to load policy '{policy_name}': {e}")
            continue
    
    if not policies:
        raise RuntimeError("No policies were successfully loaded")
    
    # Stage 2: Set up opponent inference for policies that need it
    if pending_inference_configs:
        print("\nSetting up opponent inference (Stage 2):")
        for policy_id, opponent_inference_config in pending_inference_configs.items():
            if opponent_inference_config.get('enabled', False):
                try:
                    # Lazy import to avoid circular dependency
                    from opponent_inference.inference import load_inference
                    
                    # Get the actual policy object
                    if include_metadata:
                        policy_obj = policies[policy_id]["policy"]
                    else:
                        policy_obj = policies[policy_id]
                    
                    # Create opponent inference with all available policies
                    opponent_inference = load_inference(opponent_inference_config, game)
                    
                    # Inject it into the policy (assuming we add this method)
                    if hasattr(policy_obj, 'set_opponent_inference'):
                        policy_obj.set_opponent_inference(opponent_inference)
                        print(f"  ✓ {policy_id}: Opponent inference enabled")
                    else:
                        print(f"  ⚠ {policy_id}: Policy doesn't support opponent inference")
                        
                except Exception as e:
                    print(f"  ✗ {policy_id}: Failed to set up opponent inference: {e}")
    
    return policies



def generate_cache_filename(game_config: Dict[str, Any]) -> str:
    """Generate a cache filename based on game configuration."""
    game_name = game_config["name"]
    params = game_config.get("parameters", {})
    
    if game_name == "mnk_game":
        m, n, k = params.get('m'), params.get('n'), params.get('k')
        rules = params.get('rules', {})
        
        # Determine specific variant tag
        variant_tag = "standard"
        if rules.get("allowed_directions"): 
            variant_tag = "restricted"
        if rules.get("p0_extra_k"): 
            variant_tag = "asymmetric_k"
        if rules.get("opening_moves"): 
            variant_tag = "opening_moves"
        
        return f"state_dataset/pkl/dataset_mnk_{m}x{n}_k{k}_{variant_tag}.pkl"
    
    # Default fallback for other games
    return f"state_dataset/pkl/dataset_{game_name}.pkl"

def generate_cache_filename(game_config: Dict[str, Any]) -> str:
    """Generate a cache filename based on game configuration."""
    game_name = game_config["name"]
    params = game_config.get("parameters", {})
    
    if game_name == "mnk_game":
        m, n, k = params.get('m'), params.get('n'), params.get('k')
        rules = params.get('rules', {})
        
        # Determine specific variant tag
        variant_tag = "standard"
        if rules.get("allowed_directions"): 
            variant_tag = "restricted"
        if rules.get("p0_extra_k"): 
            variant_tag = "asymmetric_k"
        if rules.get("opening_moves"): 
            variant_tag = "opening_moves"
        
        return f"state_dataset/pkl/dataset_mnk_{m}x{n}_k{k}_{variant_tag}.pkl"
    
    # Default fallback for other games
    return f"state_dataset/pkl/dataset_{game_name}.pkl" 



# def load_policies(policies_config: List[Dict[str, Any]], game: pyspiel.Game) -> Dict[str, Any]:
#     """Load and initialize policies from configuration."""
#     policies = {}
    
#     print(f"\nAvailable policies: {list(POLICY_REGISTRY.keys())}")
#     print("Loading policies:")
    
#     for i, policy_config in enumerate(policies_config):
#         try:
#             # Create a copy to inject the game instance
#             policy_config_with_game = policy_config.copy()
#             if "parameters" not in policy_config_with_game:
#                 policy_config_with_game["parameters"] = {}
#             policy_config_with_game["parameters"]["game"] = game
            
#             policy = instantiate_policy(policy_config_with_game)
#             policy_name = policy_config["name"]
            
#             # Handle duplicate names (e.g. random vs random)
#             policy_id = f"{policy_name}_{i}" if policy_name in policies else policy_name
        
#             # Store the wrapper dict that analysis.py expects
#             policies[policy_id] = {
#                 "policy": policy,
#                 "config": policy_config
#             }
#             print(f"  ✓ {policy_id}: {policy.__class__.__name__}")
            
#         except Exception as e:
#             policy_name = policy_config.get("name", "unknown")
#             print(f"  ✗ Failed to load policy '{policy_name}': {e}")
#             import traceback
#             traceback.print_exc()
#             continue
    
#     if not policies:
#         raise RuntimeError("No policies were successfully loaded")
    
#     return policies