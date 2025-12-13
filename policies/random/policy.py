from policies.base import GamePolicy
import numpy as np
import pyspiel
from typing import Dict


class RandomPolicy(GamePolicy):
    """Policy that selects actions uniformly at random."""
    
    def __init__(self, game: pyspiel.Game, seed: int = None):
        super().__init__(game)
        self.rng = np.random.RandomState(seed)
    
    def action_likelihoods(self, state: pyspiel.State) -> Dict[int, float]:
        legal_actions = state.legal_actions()
        if not legal_actions:
            return {}
        
        uniform_prob = 1.0 / len(legal_actions)
        return {action: uniform_prob for action in legal_actions}
    
    def step(self, state):
        legal_actions = state.legal_actions()
        if not legal_actions:
            return None
        return self.rng.choice(legal_actions)
