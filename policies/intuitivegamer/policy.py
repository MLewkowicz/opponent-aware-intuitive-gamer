from policies.base import GamePolicy
import numpy as np
import pyspiel
from typing import Dict

DIRECTIONS = [(1,0),(0,1),(1,1),(1,-1)]  # h,v,diag,anti

def longest_chain(board, player):
    """Return max contiguous stones for `player` on board."""
    best = 0
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols):
            if board[r, c] != player: 
                continue
            for dr, dc in DIRECTIONS:
                length = 1
                rr, cc = r + dr, c + dc
                while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == player:
                    length += 1
                    rr += dr; cc += dc
                best = max(best, length)
    return best


def softmax(x):
    e = np.exp(x - np.max(x))   # subtract max for numerical stability
    return e / np.sum(e)

class IntuitiveGamerPolicy(GamePolicy):

    # ----- helpers -----
    def _apply_action(self, state, action):
        nxt = state.clone()
        nxt.apply_action(action)
        return nxt
    
    def _extract_board(self, state):
        """Assumes OpenSpiel tic-tac-toe representation:  3×3×3 tensor, with channels [X,O,empty_or_turn]."""
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

    def _uaux(self, state: pyspiel.State, action: int) -> float:
        """Compute auxiliary utility for a given action based on auxiliary utility defined in intuitive gamer paper."""

        nrows = self.game.observation_tensor_shape()[1]
        ncols = self.game.observation_tensor_shape()[2]
        loc = np.array([action // ncols, action % ncols])  # Convert action to (row, col)

        center = np.array([(nrows - 1) / 2, (ncols - 1) / 2])
        dist_to_center = np.linalg.norm(loc - center)/(np.linalg.norm(center))    
        return dist_to_center
    
    def _uself(self, state: pyspiel.State, action: int) -> float:
        """Compute the self utility for a given action to ensure that the agent wins as quickly as possible."""
        cur_player = state.current_player()
        nxt_state = self._apply_action(state, action)
        board = self._extract_board(nxt_state)

        me = 1 if cur_player == 0 else -1

        longest_me = longest_chain(board, me)

        if nxt_state.is_terminal() and nxt_state.returns()[cur_player] > 0:
            longest_me += 1
        
        return float(longest_me)

    # ----- opponent utility -----
    def _uopp(self, state, action: int) -> float:
        cur_player = state.current_player()
        nxt = self._apply_action(state, action)
        board = self._extract_board(nxt)

        me = 1 if cur_player == 0 else -1
        opp = -me

        longest_opp = longest_chain(board, opp)

        opp_wins = nxt.is_terminal() and nxt.returns()[1-cur_player] > 0
        if opp_wins:
            return float(longest_opp)

        # otherwise subtract 0.5
        return float(longest_opp - 0.5)

    def action_likelihoods(self, state: pyspiel.State) -> Dict[int, float]:
        likelihoods = {}
        for action in state.legal_actions():
            d = self._uaux(state, action)
            n1 = self._uself(state, action)
            n2 = self._uopp(state, action)
            
            likelihoods[action] = np.power(2, (1 - d) + n1 + n2)
        
        values = np.array(list(likelihoods.values()))
        probs = softmax(values)
        for i, action in enumerate(likelihoods.keys()):
            likelihoods[action] = probs[i]

        return likelihoods