import numpy as np
import pyspiel
import random

DIRECTIONS = [(1,0),(0,1),(1,1),(1,-1)]  # h,v,diag,anti

def longest_chain(state, player, game):
    """Return max contiguous stones for `player` on board."""
    best = 0
    def extract_board(state):
        """Assumes OpenSpiel tic-tac-toe representation:  3×3×3 tensor, with channels [X,O,empty_or_turn]."""
        obs = np.array(state.observation_tensor()).reshape(game.observation_tensor_shape())
        # channel 0 = X, channel 1 = O (typical)
        X = obs[2, :, :]
        O = obs[1, :, :]
        # Represent board as 1=X, -1=O, 0=empty
        return X - O
    board = extract_board(state)
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

def generate_all_states(ds, state, visited, num_turns, game):
    key = state.to_string()
    if key in visited or state.is_terminal():
        return
    
    me = 1 if state.current_player() == 0 else -1
    opp = -me

    longest_chain_me = longest_chain(state, me, game)
    longest_chain_opp = longest_chain(state, opp, game)

    info = {
        'state': state.clone(),
        'longest_chain_me': longest_chain_me,
        'longest_chain_opp': longest_chain_opp,
        'freespace': len(state.legal_actions()),
        'winning': longest_chain_me >= longest_chain_opp,
        'current_player': me,
        'num_turns': num_turns 
    }
    visited[key] = info
    ds.add(info)

    for a in state.legal_actions():
        child = state.clone()
        child.apply_action(a)
        num_turns_child = num_turns + 1
        generate_all_states(ds, child, visited, num_turns_child, game)


class GameStateDataset:
    def __init__(self, game: pyspiel.Game):
        self._items = []    # list of dicts
        self._indices = None   # for no-replacement mode
        self.game = game
        generate_all_states(self, game.new_initial_state(), {}, 0, game)
        

    
    def add(self, info):
        """info is the dict you are currently putting in visited[key]"""
        self._items.append(info)

    # --- Query builder ----------------------------------------------------
    def filter(self, **kwargs):
        """
        Simple equality filters:
        ds.filter(winning=True)
        ds.filter(current_player=1)
        """
        out = []
        for x in self._items:
            try:
                if all(x.get(k) == v for k, v in kwargs.items()):
                    out.append(x)
            except Exception as e:
                print(f"Error filtering item {x}: {e}")
                continue
        return GameStateView(out)   # return a view you can continue filtering

    def where(self, fn):
        """
        Free-form predicate filter:
        ds.where(lambda x: x['freespace'] < 5 and x['winning'])
        """
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
            except Exception as e:
                print(f"Error filtering item {x}: {e}")
                continue
        return GameStateView(out)

    def where(self, fn):
        return GameStateView([x for x in self._items if fn(x)])

    def sample(self, k=1, replace=True):
        if not self._items:
            print("Warning: Trying to sample from empty dataset")
            return []
        if not replace and k > len(self._items):
            print(f"Warning: Cannot sample {k} items without replacement from {len(self._items)} items")
            return []
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





if __name__ == "__main__":
    game = pyspiel.load_game("tic_tac_toe")
    dataset = GameStateDataset(game)
    # populate all_states as in state_sampler/state_generation/tictactoe.py
