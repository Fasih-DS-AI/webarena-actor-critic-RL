"""Actor network — the policy pi(a|s).

This is the RL analogue of the proposal's "LLM Actor": a stochastic policy that,
given the current page state, outputs a probability distribution over the
available element-activation actions. Invalid actions (empty slots) are masked
out before forming the categorical distribution, so the policy can only select
grounded elements — matching the proposal's constraint that actions are grounded
to real DOM elements with "no hallucinated interactions".
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .networks import MLPTorso, orthogonal_init

# Large negative logit added to masked actions -> ~zero probability.
_MASK_FILL = -1e8


class Actor(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int,
                 hidden_sizes=(128, 128)):
        super().__init__()
        self.torso = MLPTorso(obs_dim, hidden_sizes)
        # Small gain on the policy head keeps the initial policy close to
        # uniform (high entropy), which aids early exploration.
        self.head = orthogonal_init(
            nn.Linear(self.torso.output_dim, num_actions), gain=0.01
        )

    def logits(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor]) -> torch.Tensor:
        logits = self.head(self.torso(obs))
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, _MASK_FILL)
        return logits

    def distribution(self, obs: torch.Tensor,
                     action_mask: Optional[torch.Tensor]) -> Categorical:
        return Categorical(logits=self.logits(obs, action_mask))

    def forward(self, obs: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> Categorical:
        return self.distribution(obs, action_mask)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, action_mask: torch.Tensor,
            greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (or greedily pick) an action; return (action, log_prob)."""
        dist = self.distribution(obs, action_mask)
        if greedy:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate_actions(self, obs: torch.Tensor, action_mask: torch.Tensor,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """For PPO updates: return (log_prob, entropy) of given actions."""
        dist = self.distribution(obs, action_mask)
        return dist.log_prob(actions), dist.entropy()
