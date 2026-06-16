"""Heuristic / static-reasoner baseline.

This models the kind of non-learning agent the proposal contrasts against (the
"static policy" — GPT-4 direct / ReAct prompting in the original WebArena paper).
It reads the page from the observation and *attempts* to ground the correct
element for the current required intent, but with a fixed ``mistake_prob`` it
mis-grounds and clicks a random valid element instead — capturing WebArena's
core failure mode (imperfect grounding, no improvement from experience).

Crucially this agent never learns: its accuracy is fixed, so over a long-horizon
task small per-step grounding errors compound into a low overall success rate,
which is precisely the behaviour the Actor-Critic method is designed to fix.
"""

from __future__ import annotations

import numpy as np

from ..env.observation import A_MAX, N_INTENTS, OBS_DIM, _SLOT_DIM
from ..env.tasks import INTENTS

_GLOBAL_REQ_OFFSET = A_MAX * _SLOT_DIM  # start of required-intent one-hot block


class GreedyAgent:
    name = "Heuristic"

    def __init__(self, mistake_prob: float = 0.45, seed: int = 0):
        self.mistake_prob = mistake_prob
        self.rng = np.random.default_rng(seed)

    def _required_intent_id(self, obs: np.ndarray) -> int:
        block = obs[_GLOBAL_REQ_OFFSET:_GLOBAL_REQ_OFFSET + N_INTENTS]
        return int(np.argmax(block))

    def _slot_intent_id(self, obs: np.ndarray, slot: int) -> int:
        base = slot * _SLOT_DIM
        return int(np.argmax(obs[base:base + N_INTENTS]))

    def act(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        valid = np.flatnonzero(action_mask)
        if len(valid) == 0:
            return 0
        # With probability mistake_prob, mis-ground -> random valid click.
        if self.rng.random() < self.mistake_prob:
            return int(self.rng.choice(valid))
        # Otherwise attempt correct grounding.
        req = self._required_intent_id(obs)
        for slot in valid:
            if self._slot_intent_id(obs, int(slot)) == req:
                return int(slot)
        return int(self.rng.choice(valid))
