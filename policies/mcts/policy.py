import sys
import os
import random
import math
import pyspiel
import numpy as np
from policies.base import GamePolicy

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        
        if state.is_terminal():
            self.untried_actions = []
        else:
            self.untried_actions = state.legal_actions()
            random.shuffle(self.untried_actions)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def ucb_score(self, child, exploration_weight):
        if child.visits == 0:
            return float('inf')
            
        # UCB1 Formula
        exploitation = child.value / child.visits
        exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
        return exploitation + exploration

    def best_child(self, exploration_weight=1.0):
        best_score = float('-inf')
        best_node = None
        for child in self.children:
            score = self.ucb_score(child, exploration_weight)
            if score > best_score:
                best_score = score
                best_node = child
        return best_node

class MCTSAgent(GamePolicy):
    def __init__(self, game, player_id=0, iterations=1000, exploration_weight=1.41, max_depth=50, stochastic=True):
        super().__init__(game)
        self.player_id = player_id
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.stochastic = stochastic
        
    def action_likelihoods(self, state):
        # MCTS Logic remains the same
        if self.stochastic: 
            for _ in range(self.iterations):
                root = MCTSNode(state=state.clone())
                # Note: We rebuild the tree every step for pure MCTS. 
                # Ideally, you keep the tree, but for this snippet, rebuilding is safer to avoid state desync.
                for _ in range(self.iterations):
                    node = self.select(root)
                    # FIX 1: Returns vector
                    rewards = self.simulate(node)
                    # FIX 2: Backpropagate vector
                    self.backpropagate(node, rewards)
                
                total_visits = sum(child.visits for child in root.children)
                # Handle edge case where root has no children (terminal state passed in)
                if total_visits == 0:
                    return {}

                likelihoods = {child.action: child.visits / total_visits for child in root.children}
                
                for a in state.legal_actions():
                    if a not in likelihoods:
                        likelihoods[a] = 0.0
                        
                return likelihoods
        else:
            action = self.step(state) # Reuse step for deterministic
            likelihoods = {a: 0.0 for a in state.legal_actions()}
            if action is not None:
                likelihoods[action] = 1.0
            return likelihoods
    
    def step(self, state):
        # We assume the tree building happens inside action_likelihoods or here. 
        # For efficiency, let's implement the tree build directly in a helper or here.
        # But to keep it consistent with your request to use `action_likelihoods`:
        
        # If we need to run MCTS for this step specifically:
        root = MCTSNode(state=state.clone())
        
        # Check simple terminal case
        if root.state.is_terminal():
            return None

        for _ in range(self.iterations):
            node = self.select(root)
            rewards = self.simulate(node)
            self.backpropagate(node, rewards)

        # For the final step, MCTS usually picks the most visited child (Robust Child)
        # regardless of stochastic/deterministic flag for the tree search part.
        if not root.children:
            return random.choice(state.legal_actions())

        # If stochastic is requested for the final policy output:
        if self.stochastic:
            total_visits = sum(c.visits for c in root.children)
            probs = [c.visits/total_visits for c in root.children]
            actions = [c.action for c in root.children]
            return np.random.choice(actions, p=probs)
        else:
            # Deterministic: Return action with max visits
            best_child = max(root.children, key=lambda c: c.visits)
            return best_child.action

    def select(self, node):
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child(self.exploration_weight)
                if node is None: 
                    break 
        return node

    def expand(self, node):
        action = node.untried_actions.pop()
        next_state = node.state.clone()
        next_state.apply_action(action)
        child_node = MCTSNode(state=next_state, parent=node, action=action)
        node.children.append(child_node)
        return child_node

    def simulate(self, node):
        """
        Returns the full returns vector for all players.
        """
        current_state = node.state.clone()
        depth = 0

        while not current_state.is_terminal() and depth < self.max_depth:
            legal_actions = current_state.legal_actions()
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            current_state.apply_action(action)
            depth += 1

        # FIX: Return the full rewards vector [p0_reward, p1_reward, ...]
        return current_state.returns()

    def backpropagate(self, node, rewards):
        """
        Updates nodes with the reward specific to the player who made the move 
        leading to that node.
        """
        while node is not None:
            node.visits += 1
            
            # The value of a node is "how good was the move that got us here?"
            # The move was made by node.parent.state.current_player()
            if node.parent is not None:
                # Identify who made the move to get to this node
                player_who_moved = node.parent.state.current_player()
                # Add that specific player's reward to this node's value
                node.value += rewards[player_who_moved]
            
            node = node.parent
            
if __name__ == "__main__":
    game_name = "connect_four" 
    game = pyspiel.load_game(game_name)
    
    agent = MCTSAgent(game, player_id=0, iterations=1000)

    print(f"Starting {game_name}...")
    state = game.new_initial_state()

    while not state.is_terminal():
        print(f"\nCurrent Player: {state.current_player()}")
        print(state)

        if state.current_player() == 0:
            print("AI Thinking...")
            action = agent.select_action(state)
            print(f"AI chose: {state.action_to_string(state.current_player(), action)}")
            state.apply_action(action)
        else:
            legal_actions = state.legal_actions()
            action = random.choice(legal_actions)
            print(f"Random Opponent chose: {state.action_to_string(state.current_player(), action)}")
            state.apply_action(action)

    print("\nGame Over!")
    print(f"Returns: {state.returns()}")