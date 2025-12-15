from policies.base import GamePolicy
import numpy as np
import pyspiel
from typing import Dict

DIRECTIONS = [(1,0),(0,1),(1,1),(1,-1)]  # h,v,diag,anti



def softmax(x):
    e = np.exp(x - np.max(x))   # subtract max for numerical stability
    return e / np.sum(e)

class IntuitiveGamerPolicy(GamePolicy):

    def __init__(self, game, weights={'connect': 0.5, 'block': 0.5, 'center': 0.5}, **kwargs):
        # print(f"[DEBUG] IntuitiveGamerPolicy.__init__ called, id={id(self)}")
        super().__init__(game)  # Call parent constructor first
        self.weights = weights
        self.directions = [(1,0), (0,1), (1,1), (1,-1)]
        self.opponent_inference = None  # Will be injected later if needed
        # print(f"[DEBUG] opponent_inference initialized to None, will be injected later")

        optimal_weights_config = kwargs.get("optimal_weights", [])
        self.optimal_weights_dict = {
            entry["name"]: entry["parameters"]
            for entry in optimal_weights_config
        }
        self.eta = kwargs.get("eta", 0.5)  # Learning rate for weight updates

    def set_opponent_inference(self, opponent_inference):
        """Inject opponent inference after policy creation to avoid circular imports."""
        self.opponent_inference = opponent_inference
        # print(f"[DEBUG] opponent_inference injected: {opponent_inference}")



    def _longest_chain_with_dirs(self, board, player_val, directions):
        """Calculates max chain for player_val restricted to specific directions."""
        best = 0
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                if board[r, c] != player_val: continue
                
                for dr, dc in directions:
                    length = 1
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == player_val:
                        length += 1
                        rr += dr; cc += dc
                    best = max(best, length)
        return best
    
    # ----- helpers -----
    def _apply_action(self, state, action):
        nxt = state.clone()
        nxt.apply_action(action)
        return nxt
    
    def _extract_board(self, state):
        # For terminal states or when we need a specific player's view
        if state.is_terminal():
            # Use player 0's observation for terminal states
            obs = np.array(state.observation_tensor(0)).reshape(self.game.observation_tensor_shape())
        else:
            # Use current player's observation
            current_player = state.current_player()
            if current_player >= 0:  # Valid player
                obs = np.array(state.observation_tensor(current_player)).reshape(self.game.observation_tensor_shape())
            else:
                # Fallback to player 0's view if current_player is invalid
                obs = np.array(state.observation_tensor(0)).reshape(self.game.observation_tensor_shape())
        
        # channel 0 = X, channel 1 = O (typical)
        X = obs[2, :, :]
        O = obs[1, :, :]
        # Represent board as 1=X, -1=O, 0=empty
        return X - O

    def _get_game_info(self, player_id):
        """Helper to safely fetch rules from the game instance."""
        # Defaults
        k = 3
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        
        if hasattr(self.game, "get_win_length"):
            k = self.game.get_win_length(player_id)
        if hasattr(self.game, "get_valid_directions"):
            dirs = self.game.get_valid_directions(player_id)
            
        return k, dirs
    
    def _uaux(self, state: pyspiel.State, action: int) -> float:
        """Compute auxiliary utility for a given action based on auxiliary utility defined in intuitive gamer paper."""

        nrows = self.game.observation_tensor_shape()[1]
        ncols = self.game.observation_tensor_shape()[2]
        loc = np.array([action // ncols, action % ncols])  # Convert action to (row, col)

        center = np.array([(nrows - 1) / 2, (ncols - 1) / 2])
        dist_to_center = np.linalg.norm(loc - center)/(np.linalg.norm(center))    
        return dist_to_center
    
    def _uself(self, state, action):
        p_id = state.current_player()
        nxt = self._apply_action(state, action)
        
        # 1. Immediate Win Check
        # Pyspiel returns are relative to players. 
        if nxt.is_terminal() and nxt.returns()[p_id] > 0:
            return 10.0 # Max reward
            
        # 2. Heuristic Progress
        # Get target K and allowed directions for ME
        target_k, valid_dirs = self._get_game_info(p_id)
        
        board = self._extract_board(state)
        shape = self.game.observation_tensor_shape()
        r, c = action // shape[2], action % shape[2]
        board[r, c] = 1
        
        longest = self._longest_chain_with_dirs(board, 1, valid_dirs)
        
        return float(min(longest, target_k))

    # ----- opponent utility -----
    def _uopp(self, state, action):
        p_id = state.current_player()
        opp_id = 1 - p_id
        
        # Get target K and allowed directions for OPPONENT
        opp_k, opp_dirs = self._get_game_info(opp_id)
        
        board = self._extract_board(state)
        shape = self.game.observation_tensor_shape()
        r, c = action // shape[2], action % shape[2]
        
        # Simulate opponent playing here
        board[r, c] = -1 
        
        longest_opp = self._longest_chain_with_dirs(board, -1, opp_dirs)
        
        # Dynamic blocking threshold: 
        # If opponent reaches their K, it's a critical block.
        if longest_opp >= opp_k:
            return float(longest_opp)
        
        return float(longest_opp - 0.5)

    def action_likelihoods(self, state: pyspiel.State) -> Dict[int, float]:
        likelihoods = {}
        for action in state.legal_actions():
            d = self._uaux(state, action)
            n1 = self._uself(state, action)
            n2 = self._uopp(state, action)
            
            val = (self.weights['center'] * (1 - d)) + \
                  (self.weights['connect'] * n1) + \
                  (self.weights['block'] * n2)
            score = np.power(2, val)

            # score = np.power(2, (1 - d) + n1 + n2)
            likelihoods[action] = score
        
        values = np.array(list(likelihoods.values()))
        probs = softmax(values)
        for i, action in enumerate(likelihoods.keys()):
            likelihoods[action] = probs[i]

        return likelihoods
    
    def step(self, state):
        if self.opponent_inference:
            # Update opponent model based on action history
            action_history = self.action_choices.get(1 - self.player_id, [])
            posterior = self.opponent_inference.calculate_likelihoods(action_history)
            # Initialize averaged weights
            avg_weights = {
                k: 0.0
                for k in next(iter(self.optimal_weights_dict.values())).keys()
            }

            # Weighted average
            for policy_name, params in self.optimal_weights_dict.items():
                p = posterior.get(policy_name, 0.0)
                for param_name, value in params.items():
                    avg_weights[param_name] += p * value

            for k in self.weights:
                self.weights[k] = (
                    (1.0 - self.eta) * self.weights[k]
                    + self.eta * avg_weights[k]
                )

        probs_dict = self.action_likelihoods(state)
        if not probs_dict:
            return None
            
        actions = list(probs_dict.keys())
        probs = list(probs_dict.values())
                
        return np.random.choice(actions, p=probs)