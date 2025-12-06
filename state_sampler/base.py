from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import pyspiel


class StateSampler(ABC):
    """
    All state samplers must return a sampled state from the game according to some specificed distribution.
    """

    def __init__(self, game: pyspiel.Game):
        super().__init__()
        self.game = game

    @abstractmethod
    def sample_state(self) -> pyspiel.State:
        pass
    