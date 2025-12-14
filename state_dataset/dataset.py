import numpy as np
import pyspiel
import random
import pickle
import os
import yaml
import argparse
from tqdm import tqdm
import sys

current_dir = os.path.dirname(os.path.abspath(__file__)) # .../state_dataset
project_root = os.path.dirname(current_dir)              # .../intuitive-gamer-memo
if project_root not in sys.path:
    sys.path.append(project_root)

# Default directions if game doesn't specify them
DEFAULT_DIRECTIONS = [(1,0), (0,1), (1,1), (1,-1)] # h, v, diag, anti

def extract_board(state, game):
    """
    Extracts board from OpenSpiel observation tensor.
    Assumes shape (3, rows, cols) with planes: [Player(X), Opponent(O), Empty].
    Returns board where: 1=CurrentPlayer, -1=Opponent, 0=Empty.
    """
    shape = game.observation_tensor_shape()
    obs = np.array(state.observation_tensor(state.current_player())).reshape(shape)
    
    # Plane 0: Current Player
    # Plane 1: Opponent 
    me_plane = obs[0, :, :]
    opp_plane = obs[1, :, :]
    
    return me_plane - opp_plane

def longest_chain(state, player_val, game):
    """
    Return max contiguous stones for `player` on board.
    player_val: 1 for current player, -1 for opponent (relative to state.current_player)
    """
    best = 0
    board = extract_board(state, game)
    rows, cols = board.shape
    
    # 1. Determine Player ID to fetch valid directions
    current_p_id = state.current_player()
    
    if player_val == 1:
        target_p_id = current_p_id
    else:
        target_p_id = 1 - current_p_id
        
    # 2. Get Valid Directions from Game (if supported)
    if hasattr(game, "get_valid_directions"):
        directions = game.get_valid_directions(target_p_id)
    else:
        directions = DEFAULT_DIRECTIONS

    # 3. Calculate Chain Length
    for r in range(rows):
        for c in range(cols):
            if board[r, c] != player_val: 
                continue
            for dr, dc in directions:
                length = 1
                rr, cc = r + dr, c + dc
                while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == player_val:
                    length += 1
                    rr += dr; cc += dc
                best = max(best, length)
    return best

def generate_all_states(ds, state, visited, num_turns, game, max_depth=None, pbar=None):
    """
    Recursively visits states. 
    Added max_depth optional parameter to prevent infinite recursion on larger boards.
    Added pbar to track progress.
    """
    key = str(state) 
    if key in visited or state.is_terminal():
        return
        
    if max_depth is not None and num_turns > max_depth:
        return
    
    # Calculate chains
    len_chain_me = longest_chain(state, 1, game)
    len_chain_opp = longest_chain(state, -1, game)

    info = {
        'state': state.clone(),
        'longest_chain_me': len_chain_me,
        'longest_chain_opp': len_chain_opp,
        'freespace': len(state.legal_actions()),
        'winning': len_chain_me > len_chain_opp, 
        'tied': len_chain_me == len_chain_opp,
        'losing': len_chain_me < len_chain_opp,
        'current_player': state.current_player(),
        'num_turns': num_turns 
    }
    
    visited[key] = info
    ds.add(info)
    
    # Update progress bar
    if pbar is not None:
        pbar.update(1)

    for a in state.legal_actions():
        child = state.clone()
        child.apply_action(a)
        generate_all_states(ds, child, visited, num_turns + 1, game, max_depth, pbar)


class GameStateDataset:
    def __init__(self, game: pyspiel.Game, max_depth_limit=None, cache_file=None):
        self._items = []    # list of dicts
        self.game = game
        self.max_depth_limit = max_depth_limit
        self.cache_file = cache_file
        
        # Try loading from cache first
        if self.cache_file and os.path.exists(self.cache_file):
            print(f"Loading cached states from {self.cache_file}...")
            if self.load():
                print(f"Loaded {len(self._items)} states from cache.")
                return

        # Start generation if no cache or load failed
        print(f"Generating states for {game}...")
        visited = {}
        
        with tqdm(desc="Generating States", unit=" states") as pbar:
            generate_all_states(self, game.new_initial_state(), visited, 0, game, max_depth=max_depth_limit, pbar=pbar)
            
        print(f"Generated {len(self._items)} unique non-terminal states.")
        
        # Save to cache if file path provided
        if self.cache_file:
            self.save()

    def add(self, info):
        """info is the dict you are currently putting in visited[key]"""
        self._items.append(info)

    def save(self):
        """Save the dataset items to a pickle file."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._items, f)
            print(f"Saved dataset to {self.cache_file}")
        except Exception as e:
            print(f"Error saving cache: {e}")

    def load(self):
        """Load the dataset items from a pickle file."""
        try:
            with open(self.cache_file, 'rb') as f:
                self._items = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False

    # --- Query builder ----------------------------------------------------
    def filter(self, **kwargs):
        out = []
        for x in self._items:
            try:
                if all(x.get(k) == v for k, v in kwargs.items()):
                    out.append(x)
            except Exception as e:
                continue
        return GameStateView(out) 

    def where(self, fn):
        return GameStateView([x for x in self._items if fn(x)])

    def all(self):
        return list(self._items)


class GameStateView:
    def __init__(self, items):
        self._items = items

    def filter(self, **kwargs):
        out = []
        for x in self._items:
            try:
                if all(x.get(k) == v for k, v in kwargs.items()):
                    out.append(x)
            except Exception:
                continue
        return GameStateView(out)

    def where(self, fn):
        return GameStateView([x for x in self._items if fn(x)])

    def sample(self, k=1, replace=True):
        if not self._items:
            print("Warning: Trying to sample from empty dataset")
            return []
        
        if not replace and k > len(self._items):
            print(f"Warning: Requesting {k} samples from {len(self._items)} items without replacement. Returning all items.")
            return random.sample(self._items, len(self._items))
            
        try:
            if replace:
                return random.choices(self._items, k=k)   # with replacement
            else:
                return random.sample(self._items, k=k)    # without replacement
        except Exception as e:
            print(f"Error during sampling: {e}")
            return []

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        yield from self._items

# --- MAIN BLOCK ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and pickle game states from config.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config file (e.g., config/baseline.yaml)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output filename. If not provided, one is generated.")
    
    args = parser.parse_args()

    try:
        # 1. Load Config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        game_cfg = config.get("game", {})
        game_name = game_cfg.get("name")
        params = game_cfg.get("parameters", {})

        # 2. Instantiate Game
        if game_name == "mnk_game":
            from games.mnk_game import MNKGame
            game = MNKGame(**params)
            print(f"Initialized Custom MNKGame: {params}")
            
            # Create a descriptive filename if none provided
            if args.output is None:
                m, n, k = params.get('m'), params.get('n'), params.get('k')
                rules = params.get('rules', {})
                
                # Determine specific variant tag
                variant_tag = "standard"
                if rules.get("allowed_directions"): variant_tag = "restricted"
                if rules.get("p0_extra_k"): variant_tag = "asymmetric_k"
                if rules.get("opening_moves"): variant_tag = "opening_moves"
                
                args.output = f"dataset_mnk_{m}x{n}_k{k}_{variant_tag}.pkl"
                
        else:
            game = pyspiel.load_game(game_name, params)
            print(f"Initialized OpenSpiel Game: {game_name}")
            if args.output is None:
                args.output = f"dataset_{game_name}.pkl"

        # 3. Run Generator
        print(f"Output will be saved to: {args.output}")
        # Note: Set max_depth_limit appropriately. For 4x4, depth 16 is fine. For 5x5, limit it or it will never finish.
        limit = 16 if params.get('m', 3) * params.get('n', 3) <= 16 else 10 
        dataset = GameStateDataset(game, max_depth_limit=limit, cache_file=args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()