"""Observation encoding for MiniWebArena.

The raw environment state is a small structured "page" (a list of UI elements,
each with an intent), plus task context. This module flattens that structure
into a fixed-size float vector suitable for an MLP policy/value network, and
produces the corresponding boolean action mask.

The encoding deliberately exposes only information an agent could read off the
page (element intents, current required subgoal, site, progress) — it never
reveals *which* element is correct, so the policy must learn the grounding
rule from reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .tasks import INTENT2ID, N_INTENTS, SITE2ID, SITES

# Maximum number of interactive element slots rendered on any page.
A_MAX = 8

# Global (non-per-slot) feature layout appended after the per-slot block:
#   required-intent one-hot (N_INTENTS)
#   site one-hot            (len(SITES))
#   progress fraction       (1)   subgoals_done / total
#   step fraction           (1)   step / max_steps
#   in-dead-end flag        (1)
_GLOBAL_DIM = N_INTENTS + len(SITES) + 3

# Per-slot features: intent one-hot (N_INTENTS) + present/valid bit (1)
_SLOT_DIM = N_INTENTS + 1

OBS_DIM = A_MAX * _SLOT_DIM + _GLOBAL_DIM
NUM_ACTIONS = A_MAX


@dataclass
class Page:
    """A rendered page: up to A_MAX element slots, each holding an intent.

    ``slot_intents[i]`` is the intent string in slot ``i`` or ``None`` if the
    slot is empty (masked out / not selectable).
    """

    slot_intents: List[Optional[str]]

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(A_MAX, dtype=bool)
        for i, intent in enumerate(self.slot_intents):
            if intent is not None:
                mask[i] = True
        return mask


def encode(
    page: Page,
    required_intent: str,
    site: str,
    progress_frac: float,
    step_frac: float,
    in_dead_end: bool,
) -> np.ndarray:
    """Encode the full observation into a fixed-size float32 vector."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    # Per-slot block.
    for i, intent in enumerate(page.slot_intents):
        base = i * _SLOT_DIM
        if intent is None:
            continue
        obs[base + INTENT2ID[intent]] = 1.0
        obs[base + N_INTENTS] = 1.0  # present/valid bit

    # Global block.
    g = A_MAX * _SLOT_DIM
    obs[g + INTENT2ID[required_intent]] = 1.0
    g2 = g + N_INTENTS
    obs[g2 + SITE2ID[site]] = 1.0
    g3 = g2 + len(SITES)
    obs[g3 + 0] = float(progress_frac)
    obs[g3 + 1] = float(step_frac)
    obs[g3 + 2] = 1.0 if in_dead_end else 0.0

    return obs
