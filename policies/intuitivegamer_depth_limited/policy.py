import random
import math
import numpy as np
from policies.base import GamePolicy
from policies.mcts.policy import MCTSNode
from policies.intuitivegamer.policy import IntuitiveGamerPolicy

class DepthLimitedIGPolicy(GamePolicy):
    def __init__(self, game, player_id=0, max_depth=3, iterations=100, exploration_weight=1.41, ig_weights=None):
        super().__init__(game)
        self.player_id = player_id
        self.max_depth = max_depth
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        
        # The internal evaluator
        self.evaluator = IntuitiveGamerPolicy(game, weights=ig_weights)

    def action_likelihoods(self, state):
        root = self._run_search(state)
        
        if not root.children:
            return {a: 1.0/len(state.legal_actions()) for a in state.legal_actions()} if state.legal_actions() else {}

        # Stochastic: Likelihoods proportional to visit counts
        total_visits = sum(child.visits for child in root.children)
        likelihoods = {}
        for child in root.children:
            likelihoods[child.action] = child.visits / total_visits
            
        return likelihoods

    def step(self, state):
        root = self._run_search(state)
        
        if not root.children:
            return np.random.choice(state.legal_actions()) if state.legal_actions() else None
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action