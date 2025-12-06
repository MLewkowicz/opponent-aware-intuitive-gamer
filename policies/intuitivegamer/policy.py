from policies.base import GamePolicy
import numpy as np
import pyspiel
from typing import Dict

class IntuitiveGamerPolicy(GamePolicy):
    def action_likelihoods(self, state: pyspiel.State) -> Dict[int, float]:
        pass