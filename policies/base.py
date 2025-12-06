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

    @abstractmethod
    def action_likelihoods(self, state: pyspiel.State) -> Dict[int, float]:
        pass