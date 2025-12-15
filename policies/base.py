from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import pyspiel


class GamePolicy(ABC):
    """
    All policies must return a likelihood over actions given a current game state.
    """

    def __init__(self, game: pyspiel.Game):
        super().__init__()
        self.game = game
        self.player_id = None  # To be set when the policy is assigned to a player
        self.action_choices = {}  # Track action choices per player

    @abstractmethod
    def action_likelihoods(self, state: pyspiel.State) -> Dict[int, float]:
        pass
    
    @abstractmethod
    def step(self, state: pyspiel.State) -> int:
        pass

    def assign_playerid(self, id: int):
        self.player_id = id


    def update_action_choices(self, action:int, state: pyspiel.State, player: int) -> int:
        self.action_choices.setdefault(player, [])
        self.action_choices[player].append((state, action))

    def reset(self):
        self.action_choices = {}