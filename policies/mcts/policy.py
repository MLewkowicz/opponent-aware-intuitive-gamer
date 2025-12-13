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
        self.game = game
        self.player_id = player_id
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.stochastic = stochastic
        
    def action_likelihoods(self, state):
        if self.stochastic: 
            action_counts = {a: 0 for a in state.legal_actions()}
            for _ in range(self.iterations):
                root = MCTSNode(state=state.clone())
                for _ in range(self.iterations):
                    node = self.select(root)
                    reward = self.simulate(node)
                    self.backpropagate(node, reward)
                
                # Calculate likelihoods based on visit counts of root's children
                total_visits = sum(child.visits for child in root.children)
                likelihoods = {child.action: child.visits / total_visits for child in root.children}
                
                # Ensure all legal actions are in the dict (even if 0 visits)
                for a in state.legal_actions():
                    if a not in likelihoods:
                        likelihoods[a] = 0.0
                        
                return likelihoods
        else:
            action = self.select_action(state)
            likelihoods = {a: 0.0 for a in state.legal_actions()}
            if action is not None:
                likelihoods[action] = 1.0
            return likelihoods
    
    def step(self, state):
        """
        Executes MCTS simulations and selects an action.
        If stochastic=True, samples based on visit count distribution.
        If stochastic=False, selects the most visited action.
        """
        # We use action_likelihoods to reuse the MCTS run logic
        probs_dict = self.action_likelihoods(state)
        
        if not probs_dict:
            return None

        actions = list(probs_dict.keys())
        probs = list(probs_dict.values())

        if self.stochastic:
            return np.random.choice(actions, p=probs)
        else:
            # Deterministic: Argmax
            return max(probs_dict, key=probs_dict.get)
    
    def select_action(self, root_state):
        # Create root node with a clone of the state
        root = MCTSNode(state=root_state.clone())

        if root.state.is_terminal():
            return None

        for _ in range(self.iterations):
            node = self.select(root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)

        # Select the best action based on pure exploitation
        best_child = root.best_child(exploration_weight=0)
        
        if best_child is None:
            return random.choice(root_state.legal_actions())
            
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
        current_state = node.state.clone()
        depth = 0

        while not current_state.is_terminal() and depth < self.max_depth:
            legal_actions = current_state.legal_actions()
            if not legal_actions:
                break
            
            action = random.choice(legal_actions)
            current_state.apply_action(action)
            depth += 1

        return current_state.returns()[self.player_id]

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
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