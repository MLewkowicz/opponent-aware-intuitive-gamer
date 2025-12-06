from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
import numpy as np

import pyspiel.Game as Game


class GamePolicy(ABC):
    """
    All policies must return a likelihood over actions given a current game state.
    """

    def __init__(self, game: Game):
        super().__init__()
        self.game = game

    @abstractmethod
    def action_likelihoods(self, state) -> np.ndarray:
        pass