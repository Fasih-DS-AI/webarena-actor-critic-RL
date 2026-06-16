"""Critic network — the state-value estimator V(s).

This is the proposal's "Critic": a lightweight, separate network that maps the
current state to a scalar estimate of expected cumulative discounted reward
("how likely is the agent to complete the task from here?"). It is trained by
regressing onto bootstrapped TD returns and provides the baseline used for
advantage estimation, which dramatically reduces policy-gradient variance.

It is a *separate* network from the Actor (not a shared torso) so that the
ablation "Actor-only, no Critic" is a clean comparison and the Critic's
contribution can be isolated, exactly as the evaluation in the proposal calls
for.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .networks import MLPTorso, orthogonal_init


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        self.torso = MLPTorso(obs_dim, hidden_sizes)
        self.head = orthogonal_init(nn.Linear(self.torso.output_dim, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return V(s) with shape (batch,)."""
        return self.head(self.torso(obs)).squeeze(-1)
