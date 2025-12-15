import numpy as np
from policies.base import GamePolicy
from policies.intuitivegamer.policy import IntuitiveGamerPolicy

class DepthLimitedIGPolicy(GamePolicy):
    """
    A Depth-Limited Intuitive Gamer that simulates play by expanding 
    only the top-k most 'sensible' actions at each level and estimating
    action values by averaging the heuristic value at the leaf nodes.
    """
    def __init__(self, game, player_id=0, max_depth=3, k=3, weights=None):
        super().__init__(game)
        self.player_id = player_id 
        self.max_depth = max_depth
        self.k = k
        
        # Internal evaluator for heuristic values (V)
        self.evaluator = IntuitiveGamerPolicy(game, weights=weights)

    def action_likelihoods(self, state):
        """
        Returns action probabilities proportional to the average value of 
        leaf nodes in the top-k search tree.
        """
        legal_actions = state.legal_actions()
        if not legal_actions:
            return {}

        action_scores = {}
        
        # Evaluate all actions at the root level (no top-k pruning at the very first step)
        # to ensure we have a distribution over all valid moves.
        for action in legal_actions:
            child_state = state.clone()
            child_state.apply_action(action)
            
            # Start recursion from depth 2 (Root is 1)
            leaf_values = self._collect_leaf_values(child_state, current_depth=2)
            
            # "Likelihood ... proportional to the average of the value function across leaf nodes"
            if leaf_values:
                avg_value = np.mean(leaf_values)
            else:
                # Fallback if no leaves (e.g. terminal immediately)
                # Use the immediate state value
                avg_value = self._get_state_value(child_state)
            
            action_scores[action] = avg_value

        # Compute Softmax over the averaged values
        # We assume values are already in the "exponential" space from Intuitive Gamer, 
        # or raw utilities?
        # The paper says: "likelihood of the next actions is then proportional to the average..."
        # If the values are utilities, we exponentiate. If they are already probabilities, we normalize.
        # IntuitiveGamer heuristic usually returns exp(score). 
        # Averaging exp(scores) is valid. 
        # We then normalize to get a PDF.
        
        total_score = sum(action_scores.values())
        if total_score == 0:
            return {a: 1.0/len(legal_actions) for a in legal_actions}
            
        return {a: score / total_score for a, score in action_scores.items()}

    def step(self, state):
        """Sample an action based on the depth-limited search distribution."""
        likelihoods = self.action_likelihoods(state)
        actions = list(likelihoods.keys())
        probs = list(likelihoods.values())
        return np.random.choice(actions, p=probs)

    def _collect_leaf_values(self, state, current_depth):
        """
        Recursively expands the tree using top-k actions and returns a list of 
        heuristic values for all leaf nodes encountered.
        """
        # Base Case: Reached max depth or terminal state
        if current_depth >= self.max_depth or state.is_terminal():
            return [self._get_state_value(state)]

        legal_actions = state.legal_actions()
        if not legal_actions:
            return [self._get_state_value(state)]

        action_heuristics = []
        for action in legal_actions:
            # We use the evaluator to get the 'local' value of this action
            val = self._get_action_heuristic(state, action)
            action_heuristics.append((val, action))
        
        # Sort descending and take top k
        action_heuristics.sort(key=lambda x: x[0], reverse=True)
        top_k_actions = [x[1] for x in action_heuristics[:self.k]]

        # 2. Expand Top-K
        all_leaves = []
        for action in top_k_actions:
            child = state.clone()
            child.apply_action(action)
            leaves = self._collect_leaf_values(child, current_depth + 1)
            all_leaves.extend(leaves)
            
        return all_leaves

    def _get_action_heuristic(self, state, action):
        """
        Helper to get the raw V(s,a) from the internal Intuitive Gamer policy.
        """
        # We access the internal scoring components of the Intuitive Gamer
        # Assuming IntuitiveGamerPolicy has `_uaux`, `_uself`, `_uopp` methods
        d = self.evaluator._uaux(state, action)
        n1 = self.evaluator._uself(state, action)
        n2 = self.evaluator._uopp(state, action)
        
        # Reconstruct value: w * U
        # Using the weights dictionary passed during init
        w = self.evaluator.weights
        val = (w['center'] * (1 - d)) + \
              (w['connect'] * n1) + \
              (w['block'] * n2)
        
        # IG uses base 2 exponentiation
        return np.power(2, val)

    def _get_state_value(self, state):
        """
        Evaluates a static board state (Leaf Node).
        """
        # 1. Check if the game is actually over
        if state.is_terminal():
            ret = state.returns()[self.player_id]
            return np.power(2, ret * 10.0)

        legal_actions = state.legal_actions()
        if not legal_actions:
            return 0.0 # Or np.power(2, 0) = 1.0 depending on preference
            
        # Max over available actions (Greedy evaluation at leaf)
        best_val = -float('inf')
        for action in legal_actions:
            val = self._get_action_heuristic(state, action)
            if val > best_val:
                best_val = val
        
        # Apply the "Suicidal Agent" fix from the previous step if not already present
        # Invert value if it's the opponent's turn at this leaf
        if state.current_player() != self.player_id:
             return 1.0 / (best_val + 1e-9)

        return best_val