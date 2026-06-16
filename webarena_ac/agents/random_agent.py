"""Random baseline: uniformly samples among valid (grounded) actions.

This is the lower-bound reference — it performs no reasoning and no learning,
analogous to an agent with no useful policy. Its success rate quantifies the
difficulty of the task distribution.
"""

from __future__ import annotations

import numpy as np


class RandomAgent:
    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, action_mask):
        valid = np.flatnonzero(action_mask)
        return int(self.rng.choice(valid)) if len(valid) else 0
